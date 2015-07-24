##############################################################################
##
## Models with Interactive Fixed Effect (Bai 2009)
##
##############################################################################

function fit(m::PanelFactorModel, f::Formula, df::AbstractDataFrame; method::Symbol = :l_bfgs, lambda::Real = 0.0, subset::Union(AbstractVector{Bool}, Nothing) = nothing, weight::Union(Symbol, Nothing) = nothing, maxiter::Integer = 10000, tol::Real = 1e-9)

	# initial check
	if method == :svd
		weight == nothing || error("The svd methods does not work with weights")
		lambda == 0.0 || error("The svd method only works with lambda = 0.0")
	elseif method == :gs
		lambda == 0.0 || error("The Gauss-Seidel method only works with lambda = 0.0")
	end

	rf = deepcopy(f)

	## decompose formula into normal  vs absorbpart
	(rf, has_absorb, absorb_formula) = decompose_absorb!(rf)
	if has_absorb
		absorb_vars = allvars(absorb_formula)
		absorb_terms = Terms(absorb_formula)
	else
		absorb_vars = Symbol[]
	end

	rt = Terms(rf)

	## create a dataframe without missing values & negative weights
	factor_vars = [m.id, m.time]
	vars = allvars(rf)
	all_vars = vcat(vars, absorb_vars, factor_vars)
	all_vars = unique(convert(Vector{Symbol}, all_vars))
	esample = complete_cases(df[all_vars])
	if weight != nothing
		esample &= isnaorneg(df[weight])
		all_vars = unique(vcat(all_vars, weight))
	end
	subdf = df[esample, all_vars]
	all_except_absorb_vars = unique(convert(Vector{Symbol}, vcat(vars, factor_vars)))
	for v in all_except_absorb_vars
		dropUnusedLevels!(subdf[v])
	end

	## create weight vector
	if weight == nothing
		w = fill(one(Float64), size(subdf, 1))
		sqrtw = w
	else
		w = convert(Vector{Float64}, subdf[weight])
		sqrtw = sqrt(w)
	end

	## Compute factors, an array of AbtractFixedEffects
	if has_absorb
		factors = AbstractFixedEffect[FixedEffect(subdf, a, sqrtw) for a in absorb_terms.terms]
		# in case some FixedEffect is aFixedEffectIntercept, remove the intercept
		if any([typeof(f) <: FixedEffectIntercept for f in factors]) 
			rt.intercept = false
		end
	end

	id = subdf[m.id]
	time = subdf[m.time]

	## Compute demeaned X
	mf = simpleModelFrame(subdf, rt, esample)
	coef_names = coefnames(mf)
	X = ModelMatrix(mf).m
	if weight != nothing
		broadcast!(*, X, X, sqrtw)
	end
	if has_absorb
		(X, iterations, converged) = demean!(X, factors; maxiter = maxiter, tol = tol)
	end


	## Compute demeaned y
	py = model_response(mf)[:]
	if eltype(py) != Float64
		y = convert(py, Float64)
	else
		y = py
	end
	if weight != nothing
		broadcast!(*, y, y, sqrtw)
	end
	if has_absorb
		(y, iterations, converged) = demean!(y, factors)
	end


	## Compute demeaned X
	if allvars(rf.rhs) != []
		# initial b
		crossx = cholfact!(At_mul_B(X, X))
		b =  crossx \ At_mul_B(X, y)

		# dispatch manually to the right method
		if method == :svd
			M = crossx \ X'
			fit_svd(X, M, b, y, id, time, m.rank, maxiter, tol) 
		elseif method == :gs
			M = crossx \ X'
			fit_reg(X, M, b, y, id, time, sqrtw, m.rank, maxiter, tol) 
		else
			fit_optimization(X, b, y, id, time, m.rank, method, lambda, sqrtw, maxiter, tol) 
		end
	else
		if method == :svd
			fit_svd(y, id, time, m.rank, maxiter = maxiter, tol = tol)
		elseif method == :backpropagation
		    fit_backpropagation(y, id, time, m.rank, sqrtw, regularizer = 0.001, learning_rate = 0.001, maxiter = maxiter, tol = tol)
		else
		    fit_optimization(y, id, time, m.rank, sqrtw, lambda = lambda, method = method, maxiter = maxiter, tol = tol)
		end
	end
end


##############################################################################
##
## Estimate Model by gauss-seidel method
##
##############################################################################

function fit_reg{Tid, Rid, Ttime, Rtime}(X::Matrix{Float64}, M::Matrix{Float64}, b::Vector{Float64}, y::Vector{Float64}, id::PooledDataVector{Tid, Rid}, time::PooledDataVector{Ttime, Rtime}, sqrtw::Vector{Float64}, rank::Integer, maxiter::Integer, tol::Real)

	# initialize
	idf = PooledFactor(id.refs, length(id.pool), rank)
	timef = PooledFactor(time.refs, length(time.pool), rank)

	res = y - X*b
	result = fit_optimization(res, id, time, rank, sqrtw, lambda = 0.0,  method = :gradient_descent, tol = 0.01)
	copy!(idf.pool, result.loadings)
	copy!(timef.pool, result.factors)

	new_b = deepcopy(b)
	scaledloadings = similar(idf.pool)
	scaledfactors = similar(timef.pool)

	# starts loop
	converged = false
	iterations = maxiter
	iter = 0
	while iter < maxiter
		iter += 1
		(new_b, b) = (b, new_b)

		# Given beta, compute incrementally an approximate factor model
		subtract_b!(res, y, b, X)
		for r in 1:rank
			update!(idf, timef, res, sqrtw, r)
			subtract_factor!(res, sqrtw, idf, timef, r)
			rescale!(idf, timef, r)
		end

		# Given factor model, compute beta
		subtract_factor!(res, y, sqrtw, idf, timef)
		new_b = M * res

		# Check convergence
		error = chebyshev(new_b, b)
		if error < tol 
			converged = true
			iterations = iter
			break
		end
	end

	rescale!(scaledloadings, scaledfactors, idf.pool, timef.pool)
	return PanelFactorModelResult(b, id, time, scaledloadings, scaledfactors, iterations, converged)
end


##############################################################################
##
## Estimate Model by optimization routine
##
##############################################################################

function fit_optimization{Tid, Rid, Ttime, Rtime}(X::Matrix{Float64},  b::Vector{Float64}, y::Vector{Float64}, id::PooledDataVector{Tid, Rid}, time::PooledDataVector{Ttime, Rtime}, rank::Integer, method::Symbol, lambda::Real, sqrtw::Vector{Float64}, maxiter::Int, tol::Real)

	n_regressors = size(X, 2)

	# initialize a factor model. Requires a good differenciation accross dimensions from the start
	res = y - X*b
	result = fit_optimization(res, id, time, rank, sqrtw, lambda = lambda,  method = :gradient_descent, tol = 0.01)
	loadings = result.loadings
	factors = result.factors

	# squeeze (b, loadings and factors) into a vector x0
	x0 = Array(Float64, n_regressors + rank * length(id.pool) + rank * length(time.pool))
	x0[1:n_regressors] = b
	fill!(x0, loadings,  n_regressors)
	fill!(x0, factors,  n_regressors + length(id.pool) * rank)

	# translate indexes
	idrefs = similar(id.refs)
	@inbounds for i in 1:length(id.refs)
		idrefs[i] = n_regressors + (id.refs[i] - 1) * rank 
	end
	timerefs = similar(time.refs)
	@inbounds for i in 1:length(time.refs)
		timerefs[i] = n_regressors + length(id.pool) * rank + (time.refs[i] - 1) * rank 
	end

	# use Xt rather than X (cache performance)
	Xt = X'

	# optimize
	f = x -> sum_of_squares(x, sqrtw, y, timerefs, idrefs, n_regressors, rank, Xt, lambda)
	g! = (x, storage) -> sum_of_squares_gradient!(x, storage, sqrtw, y, timerefs, idrefs, n_regressors, rank, Xt, lambda)
	fg! = (x, storage) -> sum_of_squares_and_gradient!(x, storage, sqrtw, y, timerefs, idrefs, n_regressors, rank, Xt, lambda)
    d = DifferentiableFunction(f, g!, fg!)
    result = optimize(d, x0, method = method, iterations = maxiter, xtol = tol, ftol = 1e-32, grtol = 1e-32)  
    minimizer = result.minimum
    iterations = result.iterations
	converged =  result.x_converged || result.f_converged || result.gr_converged

	# expand minimumm -> (b, loadings and factors)
    b = minimizer[1:n_regressors]
    fill!(loadings, minimizer, n_regressors)
    fill!(factors, minimizer, n_regressors + length(id.pool) * rank)  

    # rescale factors and loadings so that factors' * factors = Id
    (loadings, factors) = rescale(loadings, factors)

    return PanelFactorModelResult(b, id, time, loadings, factors, iterations, converged)
end

function sum_of_squares{Tid, Ttime}(x::Vector{Float64}, sqrtw::Vector{Float64}, y::Vector{Float64}, timerefs::Vector{Ttime}, idrefs::Vector{Tid}, n_regressors::Integer, rank::Integer, Xt::Matrix{Float64}, lambda::Real)
	out = zero(Float64)

	# residual term
	@inbounds @simd for i in 1:length(y)
		prediction = zero(Float64)
		id = idrefs[i]
		time = timerefs[i]
		loading = x[id]
		factor = x[time]
		sqrtwi = sqrtw[i]
		for k in 1:n_regressors
		    prediction += x[k] * Xt[k, i]
		end
		for r in 1:rank
		  prediction += sqrtwi * x[id + r] * x[time + r]
		end
		error = y[i] - prediction
		out += abs2(error)
	end

	# Tikhonov term
	@inbounds @simd for i in (n_regressors+1):length(x)
	    out += lambda * abs2(x[i])
	end
	return out
end

# gradient
function sum_of_squares_gradient!{Tid, Ttime}(x::Vector{Float64}, storage::Vector{Float64}, sqrtw::Vector{Float64}, y::Vector{Float64}, timerefs::Vector{Ttime}, idrefs::Vector{Tid}, n_regressors::Integer, rank::Integer, Xt::Matrix{Float64}, lambda::Real)
    out = zero(Float64)
    fill!(storage, zero(Float64))

    # residual term
    @inbounds @simd for i in 1:length(y)
    	prediction = zero(Float64)
        idi = idrefs[i]
        timei = timerefs[i]
        sqrtwi = sqrtw[i]
        for k in  1:n_regressors
            prediction += x[k] * Xt[k, i]
        end
	    for r in 1:rank
	    	prediction += sqrtwi * x[id + r] * x[timei + r]
	    end
        error =  y[i] - sqrtwi * prediction
        for k in 1:n_regressors
            storage[k] -= 2.0 * error * Xt[k, i] 
        end
        for r in 1:rank
        	storage[timei + r] -= 2.0 * error * sqrtwi * x[idi + r] 
        end
        for r in 1:rank
        	storage[idi + r] -= 2.0 * error * sqrtwi * x[timei + r] 
        end
    end

    # Tikhonov term
    @inbounds @simd for i in (n_regressors+1):length(x)
        storage[i] += 2.0 * lambda * x[i]
    end
    return storage
end

# function + gradient in the same loop
function sum_of_squares_and_gradient!{Tid, Ttime}(x::Vector{Float64}, storage::Vector{Float64}, sqrtw::Vector{Float64}, y::Vector{Float64}, timerefs::Vector{Ttime}, idrefs::Vector{Tid}, n_regressors::Integer, rank::Integer, Xt::Matrix{Float64}, lambda::Real)
    fill!(storage, zero(Float64))
    out = zero(Float64)

    # residual term
    @inbounds @simd for i in 1:length(y)
	    prediction = zero(Float64)
		idi = idrefs[i]
		timei = timerefs[i]
		sqrtwi = sqrtw[i]
		for k in 1:n_regressors
		    prediction += x[k] * Xt[k, i]
		end
		for r in 1:rank
			prediction += sqrtwi * x[idi + r] * x[timei + r]
		end
		error = y[i] - prediction
		out += abs2(error)
		for k in 1:n_regressors
			storage[k] -= 2.0 * error  * Xt[k, i] 
		end
		for r in 1:rank
			storage[timei + r] -= 2.0 * error * sqrtwi * x[idi + r] 
		end
		for r in 1:rank
			storage[idi + r] -= 2.0 * error * sqrtwi * x[timei + r] 
		end
    end

    # Tikhonov term
    @inbounds @simd for i in (n_regressors+1):length(x)
        out += lambda * abs2(x[i])
        storage[i] += 2.0 * lambda * x[i]
    end
    return out
end



##############################################################################
##
## Estimate model by original Bai (2009) method: SVD / beta iteration
##
##############################################################################


function fit_svd{Tid, Rid, Ttime, Rtime}(X::Matrix{Float64}, M::Matrix{Float64}, b::Vector{Float64}, y::Vector{Float64}, id::PooledDataVector{Tid, Rid}, time::PooledDataVector{Ttime, Rtime}, rank::Integer, maxiter::Integer, tol::Real)
	b = M * y
	new_b = deepcopy(b)
	res_vector = Array(Float64, length(y))

	# initialize at zero for missing values
	res_matrix = fill(zero(Float64), (length(id.pool), length(time.pool)))
	predict_matrix = deepcopy(res_matrix)
	factors = Array(Float64, (length(time.pool), rank))
	variance = Array(Float64, (length(time.pool), length(time.pool)))
	converged = false
	iterations = maxiter

	# starts the loop
	iter = 0
	while iter < maxiter
		iter += 1
		(new_b, b) = (b, new_b)
		(predict_matrix, res_matrix) = (res_matrix, predict_matrix)

		# Given beta, compute the factors
		subtract_b!(res_vector, y, b, X)
		# transform vector into matrix 
		fill!(res_matrix, res_vector, id.refs, time.refs)
		# svd of res_matrix
		At_mul_B!(variance, res_matrix, res_matrix)
		F = eigfact!(Symmetric(variance), (length(time.pool) - rank + 1):length(time.pool))
		factors = F[:vectors]

		# Given the factors, compute beta
		A_mul_Bt!(variance, factors, factors)
		A_mul_B!(predict_matrix, res_matrix, variance)
		fill!(res_vector, predict_matrix, id.refs, time.refs)
		new_b = M * (y - res_vector)

		# Check convergence
		error = chebyshev(new_b, b)
		if error < tol 
			converged = true
			iterations = iter
			break
		end
	end
	newfactors = Array(Float64, (length(time.pool), rank))
	for j in 1:rank
		newfactors[:, j] = factors[:, rank + 1 - j]
	end
	loadings = predict_matrix * factors
	return PanelFactorModelResult(b, id, time, loadings, newfactors, iterations, converged)
end






