##############################################################################
##
## Models with Interactive Fixed Effect (Bai 2009)
##
##############################################################################

function fit(m::PanelFactorModel, f::Formula, df::AbstractDataFrame; method::Symbol = :bfgs, lambda::Real = 0.0, subset::Union(AbstractVector{Bool}, Nothing) = nothing, weight::Union(Symbol, Nothing) = nothing, maxiter::Integer = 10000, tol::Real = 1e-9)

	if method == :svd
		weight == nothing || error("The svd methods does not support weights")
		lambda == 0.0 || error("The svd method does not support lambda")
	end

	#################
	# Prepare the data (transform dataframe to matrix, demean if fixed effects, multiply by weight
	#################
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
	if has_absorb
		rt.intercept = false
	end

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
		factors = FixedEffect(subdf, absorb_terms.terms, sqrtw)
	end

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


	#################
	# Do the loop that estimates jointly (beta, factors, loadings)
	#################

	H = At_mul_B(X, X)
	b = H \ At_mul_B(X, y)

	if method == :svd
		M = A_mul_Bt(inv(cholfact!(H)), X)
		estimate_factor_model(X, M, b, y, subdf[m.id], subdf[m.time], m.rank, maxiter, tol) 
	else
		estimate_factor_model(X, b, y, df[m.id], df[m.time], m.rank, method, lambda, sqrtw, maxiter, tol) 
	end
end



##############################################################################
##
## Bai (2009) method: Factor-loading / beta iteration
##
##############################################################################


function Base.fill!{Tid, Ttime}(res_matrix::Matrix{Float64}, y::Vector{Float64}, res_vector::Vector{Float64}, idsrefs::Vector{Tid}, timesrefs::Vector{Ttime})
	@inbounds @simd for i in 1:length(y)
		res_matrix[idsrefs[i], timesrefs[i]] = y[i] - res_vector[i]
	end
end

function Base.fill!{Tid, Ttime}(res_vector::Vector{Float64}, y::Vector{Float64}, res_matrix::Matrix{Float64},  idsrefs::Vector{Tid}, timesrefs::Vector{Ttime})
	@inbounds @simd for i in 1:length(y)
		res_vector[i] = y[i] - res_matrix[idsrefs[i], timesrefs[i]]
	end
end

function estimate_factor_model{Tid, Rid, Ttime, Rtime}(X::Matrix{Float64}, M::Matrix{Float64}, b::Vector{Float64}, y::Vector{Float64}, ids::PooledDataVector{Tid, Rid}, times::PooledDataVector{Ttime, Rtime}, d::Integer, maxiter::Integer, tol::Real)
	b = M * y
	new_b = deepcopy(b)
	res_vector = Array(Float64, length(y))
	# initialize at zero for missing values
	res_matrix = fill(zero(Float64), (length(ids.pool), length(times.pool)))
	new_res_matrix = deepcopy(res_matrix)
	factors = Array(Float64, (length(times.pool), d))
	variance = Array(Float64, (length(times.pool), length(times.pool)))
	converged = false
	iterations = maxiter
	iter = 0
	while iter < maxiter
		iter += 1
		(new_b, b) = (b, new_b)
		(new_res_matrix, res_matrix) = (res_matrix, new_res_matrix)
		A_mul_B!(res_vector, X, b)
		# res_vector -> res_matrix
		fill!(res_matrix, y, res_vector, ids.refs, times.refs)
		# create covariance matrix and do PCA
		At_mul_B!(variance, res_matrix, res_matrix)
		F = eigfact!(Symmetric(variance), (length(times.pool) - d + 1):length(times.pool))
		# obtain d largest eigenvectors
		factors = F[:vectors]
		# compute the low rank approximation of res_matrix
		A_mul_Bt!(variance, factors, factors)
		A_mul_B!(new_res_matrix, res_matrix, variance)
		# new_res_matrix -> res_vector
		fill!(res_vector, y, new_res_matrix, ids.refs, times.refs)
		new_b = M * res_vector
		error = chebyshev(new_b, b)
		if error < tol 
			converged = true
			iterations = iter
			break
		end
	end
	newfactors = Array(Float64, (length(times.pool), d))
	for j in 1:d
		newfactors[:, j] = factors[:, d + 1 - j]
	end
	loadings = new_res_matrix * factors
	return PanelFactorModelResult(b, ids, times, loadings, newfactors, iterations, converged, false, false)
end



##############################################################################
##
## gradient implementation
##
##############################################################################


function estimate_factor_model{Tid, Rid, Ttime, Rtime}(X::Matrix{Float64},  b::Vector{Float64}, y::Vector{Float64}, ids::PooledDataVector{Tid, Rid}, times::PooledDataVector{Ttime, Rtime}, rank::Integer, method::Symbol, lambda::Real, sqrtw::Vector{Float64}, maxiter::Int, tol::Real)

	
	n_regressors = size(X, 2)

	# x is a vector composed of k coefficients, N x r loadings, T x r factors
	x0 = fill(0.1, n_regressors + rank * length(ids.pool) + rank * length(times.pool))
	x0[1:n_regressors] = b
	
	# translate ids.refs from id.pools indexes to x indexes
	idsrefs = similar(ids.refs)
	@inbounds for i in 1:length(ids.refs)
		idsrefs[i] = n_regressors + (ids.refs[i] - 1) * rank 
	end
	
	# translate time.refs from id.pools indexes to x indexes
	timesrefs = similar(times.refs)
	@inbounds for i in 1:length(times.refs)
		timesrefs[i] = n_regressors + length(ids.pool) * rank + (times.refs[i] - 1) * rank 
	end

	# Translate X (for cache property)
	Xt = X'
	# optimize
	f = x -> sum_of_squares(x, sqrtw, y, timesrefs, idsrefs, n_regressors, rank, Xt, lambda)
	g! = (x, storage) -> sum_of_squares_gradient!(x, storage, sqrtw, y, timesrefs, idsrefs, n_regressors, rank, Xt, lambda)
	fg! = (x, storage) -> sum_of_squares_and_gradient!(x, storage, sqrtw, y, timesrefs, idsrefs, n_regressors, rank, Xt, lambda)
	h =  (x, storage) -> sum_of_squares_hessian!(x, storage, sqrtw, y, timesrefs, idsrefs, n_regressors, rank, Xt, lambda)
    d = TwiceDifferentiableFunction(f, g!, fg!, h)

    result = optimize(d, x0, method = method, iterations = maxiter, xtol = tol, ftol = 1e-32, grtol = 1e-32)  
    b = result.minimum[1:n_regressors]
    # construct loadings
    loadings = Array(Float64, (length(ids.pool), rank))
    fill!(loadings, result.minimum, n_regressors)
    # construct factors
    factors = Array(Float64, (length(times.pool), rank))
    fill!(factors, result.minimum, n_regressors + length(ids.pool) * rank)  
    # normalize so that factors' * factors = Id
    (loadings, factors) = normalize(loadings, factors)
    result = PanelFactorModelResult(b, ids, times, loadings, factors, result.iterations, result.x_converged, result.f_converged, result.gr_converged)
    return result
end

function Base.fill!(M::Matrix{Float64}, v::Vector{Float64}, start::Integer)
	idx = start
	for i in 1:size(M, 1)
		for j in 1:size(M, 2)
			idx += 1
			M[i, j] = v[idx]
		end
	end
end

function normalize(loadings::Matrix{Float64}, factors::Matrix{Float64})
    U = eigfact!(At_mul_B(factors, factors))
    sqrtDx = diagm(sqrt(abs(U[:values])))
    newfactors = loadings *  U[:vectors] * sqrtDx
    V = eigfact!(At_mul_B(newfactors, newfactors))
    return ((loadings * U[:vectors]) * (sqrtDx * V[:vectors]),  factors *  U[:vectors] * (sqrtDx \ V[:vectors]))
end


##############################################################################
##
## function to optimize, gradient, hessian
##
##############################################################################


function sum_of_squares{Tid, Ttime}(x::Vector{Float64}, sqrtw::Vector{Float64}, y::Vector{Float64}, timesrefs::Vector{Ttime}, idsrefs::Vector{Tid}, n_regressors::Integer, rank::Integer, Xt::Matrix{Float64}, lambda::Real)
	out = zero(Float64)
	@inbounds @simd for i in 1:length(y)
		prediction = zero(Float64)
		id = idsrefs[i]
		time = timesrefs[i]
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
	# penalty term
	@inbounds @simd for i in (n_regressors+1):length(x)
	    out += lambda * abs2(x[i])
	end
	return out
end

# gradient
function sum_of_squares_gradient!{Tid, Ttime}(x::Vector{Float64}, storage::Vector{Float64}, sqrtw::Vector{Float64}, y::Vector{Float64}, timesrefs::Vector{Ttime}, idsrefs::Vector{Tid}, n_regressors::Integer, rank::Integer, Xt::Matrix{Float64}, lambda::Real)
    out = zero(Float64)
    fill!(storage, zero(Float64))
    @inbounds @simd for i in 1:length(y)
    	prediction = zero(Float64)
        id = idsrefs[i]
        time = timesrefs[i]
        sqrtwi = sqrtw[i]
        for k in  1:n_regressors
            prediction += x[k] * Xt[k, i]
        end
	    for r in 1:rank
	    	prediction += sqrtwi * x[id + r] * x[time + r]
	    end
        error =  y[i] - sqrtwi * prediction
        for k in 1:n_regressors
            storage[k] -= 2.0 * error * Xt[k, i] 
        end
        for r in 1:rank
        	storage[time + r] -= 2.0 * error * sqrtwi * x[id + r] 
        end
        for r in 1:rank
        	storage[id + r] -= 2.0 * error * sqrtwi * x[time + r] 
        end
    end
    # penalty term
    @inbounds @simd for i in (n_regressors+1):length(x)
        storage[i] += 2.0 * lambda * x[i]
    end
    return storage
end

# function + gradient in the same loop
function sum_of_squares_and_gradient!{Tid, Ttime}(x::Vector{Float64}, storage::Vector{Float64}, sqrtw::Vector{Float64}, y::Vector{Float64}, timesrefs::Vector{Ttime}, idsrefs::Vector{Tid}, n_regressors::Integer, rank::Integer, Xt::Matrix{Float64}, lambda::Real)
    fill!(storage, zero(Float64))
    out = zero(Float64)
    @inbounds @simd for i in 1:length(y)
	    prediction = zero(Float64)
		id = idsrefs[i]
		time = timesrefs[i]
		sqrtwi = sqrtw[i]
		for k in 1:n_regressors
		    prediction += x[k] * Xt[k, i]
		end
		for r in 1:rank
			prediction += sqrtwi * x[id + r] * x[time + r]
		end
		error = y[i] - prediction
		out += abs2(error)
		for k in 1:n_regressors
			storage[k] -= 2.0 * error  * Xt[k, i] 
		end
		for r in 1:rank
			storage[time + r] -= 2.0 * error * sqrtwi * x[id + r] 
		end
		for r in 1:rank
			storage[id + r] -= 2.0 * error * sqrtwi * x[time + r] 
		end
    end
    # penalty term
    @inbounds @simd for i in (n_regressors+1):length(x)
        out += lambda * abs2(x[i])
        storage[i] += 2.0 * lambda * x[i]
    end
    return out
end


# hessian (used in newton method)
function sum_of_squares_hessian!{Tid, Ttime}(x::Vector{Float64}, storage::Matrix{Float64}, sqrtw::Vector{Float64}, y::Vector{Float64}, timesrefs::Vector{Ttime}, idsrefs::Vector{Tid}, n_regressors::Integer, rank::Integer, Xt::Matrix{Float64}, lambda::Real)

    fill!(storage, zero(Float64))
    @inbounds @simd for i in 1:length(y)
        id = idsrefs[i]
	    time = timesrefs[i]
	    sqrtwi = sqrtw[i]
	    wi = sqrtwi^2
	    
	    prediction = zero(Float64)
	    for k in 1:n_regressors
	        prediction += x[k] * Xt[k, i]
	    end
	    for r in 1:rank
	    	prediction += sqrtwi * x[id + r] * x[time + r]
	    end
	    error = y[i] - prediction

	    # derivative wrt beta
	    for k in 1:n_regressors
	    	xk = Xt[k, i]
	    	for l in 1:k
		    	cross = 2.0 *  xk * Xt[l, i]
    			storage[k, l] += cross
    			if k != l
	    			storage[l, k] += cross
	    		end
    		end
			for r in 1:rank
		    	cross = 2.0 * sqrtwi * x[time + r] * xk
		    	storage[k, id + r] += cross
		    	storage[id + r, k] += cross
		    end
    		for r in 1:rank
		    	cross = 2.0 * sqrtwi * x[id + r] * xk
		    	storage[k, time + r] += cross
		    	storage[time + r, k] += cross
			end
	    end

        # derivative wrt loadings
        for r in 1:rank
        	for s in 1:r
    	    	cross = 2.0 * wi * x[time + r] * x[time + s]
    	    	storage[id + r, id + s] += cross
    	    	if s != r
    		    	storage[id + s, id + r] += cross
    		    end
    		end
	        for s in 1:rank
	        	cross = 2.0 * wi * x[time + r] * x[id + s]
	        	if s == r
	        		cross -= 2.0 * sqrtwi * error 
	        	end
		        storage[id + r, time + s] += cross
	        	storage[time + s, id + r] += cross
	        end
        end

	    # derivative wrt factors
	    for r in 1:rank
	    	for s in 1:r
		    	cross =  2.0 * wi * x[id + r] * x[id + s]
		    	storage[time + r, time + s] += cross
		    	if s != r
		    		storage[time + s, time + r] += cross
		    	end
	    	end
	    end
    end
    # penalty term
    @inbounds @simd for i in (n_regressors+1):length(x)
        storage[i, i] += 2.0 * lambda
    end
    return storage
end






