
##############################################################################
##
## Estimate linear factor model by gauss-seidel method
##
##############################################################################

function fit_gs{Tid, Rid, Ttime, Rtime}(X::Matrix{Float64}, M::Matrix{Float64}, b::Vector{Float64}, y::Vector{Float64}, id::PooledDataVector{Tid, Rid}, time::PooledDataVector{Ttime, Rtime}, rank::Integer, sqrtw::AbstractVector{Float64}; maxiter::Integer = 100_000, tol::Real = 1e-9)

	# initialize
	idf = PooledFactor(id.refs, length(id.pool), rank)
	timef = PooledFactor(time.refs, length(time.pool), rank)

	(idf.pool, timef.pool, iterations, converged) = fit_gs(y - X*b, id, time, rank, sqrtw, maxiter = maxiter, tol = 100 * tol)
	res = deepcopy(y)

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

		if mod(iter, 100) == 0
			rescale!(scaledloadings, scaledfactors, idf.pool, timef.pool)
			copy!(idf.pool, scaledloadings)
			copy!(timef.pool, scaledfactors)
		end
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

	(scaledloadings, scaledfactors) = rescale(idf.pool, timef.pool)
	return (b, scaledloadings, scaledfactors, [iterations], [converged])
end


##############################################################################
##
## Estimate linear factor model by optimization routine
##
##############################################################################

function fit_optimization{Tid, Rid, Ttime, Rtime}(X::Matrix{Float64},  b::Vector{Float64}, y::Vector{Float64}, id::PooledDataVector{Tid, Rid}, time::PooledDataVector{Ttime, Rtime}, rank::Integer, sqrtw::AbstractVector{Float64}; method::Symbol = :bfgs, lambda::Real = 0.0, maxiter::Integer = 100_000, tol::Real = 1e-9)

	n_regressors = size(X, 2)
	invlen = 1 / abs2(norm(sqrtw, 2)) 

	# initialize a factor model. Requires a good differenciation accross dimensions from the start
	(loadings, factors, iterations, converged) = fit_optimization(y - X*b, id, time, rank, sqrtw, lambda = lambda,  method = method, tol = 100 * tol)
	res = deepcopy(y)

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
	f = x -> sum_of_squares(x, sqrtw, y, timerefs, idrefs, n_regressors, rank, Xt, lambda, invlen)
	g! = (x, storage) -> sum_of_squares_gradient!(x, storage, sqrtw, y, timerefs, idrefs, n_regressors, rank, Xt, lambda, invlen)
	fg! = (x, storage) -> sum_of_squares_and_gradient!(x, storage, sqrtw, y, timerefs, idrefs, n_regressors, rank, Xt, lambda, invlen)
    d = DifferentiableFunction(f, g!, fg!)
    # convergence is chebyshev for x
    result = optimize(d, x0, method = method, iterations = maxiter, xtol = tol, ftol = -nextfloat(0.0), grtol = -nextfloat(0.0))
    minimizer = result.minimum
    iterations = result.iterations
	converged =  result.x_converged || result.f_converged || result.gr_converged

	# expand minimumm -> (b, loadings and factors)
    b = minimizer[1:n_regressors]
    fill!(loadings, minimizer, n_regressors)
    fill!(factors, minimizer, n_regressors + length(id.pool) * rank)  

    # rescale factors and loadings so that factors' * factors = Id
    (scaledloadings, scaledfactors) = rescale(loadings, factors)
    return (b, scaledloadings, scaledfactors, [iterations], [converged])
end

function sum_of_squares{Tid, Ttime}(x::Vector{Float64}, sqrtw::AbstractVector{Float64}, y::Vector{Float64}, timerefs::Vector{Ttime}, idrefs::Vector{Tid}, n_regressors::Integer, rank::Integer, Xt::Matrix{Float64}, lambda::Real, invlen::Real)
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
	out *= invlen

	# Tikhonov term
	@inbounds @simd for i in (n_regressors+1):length(x)
	    out += lambda * abs2(x[i])
	end
	return out
end

# gradient
function sum_of_squares_gradient!{Tid, Ttime}(x::Vector{Float64}, storage::Vector{Float64}, sqrtw::AbstractVector{Float64}, y::Vector{Float64}, timerefs::Vector{Ttime}, idrefs::Vector{Tid}, n_regressors::Integer, rank::Integer, Xt::Matrix{Float64}, lambda::Real, invlen::Real)
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
        error =  y[i] - prediction
        for k in 1:n_regressors
            storage[k] -= 2.0 * error * Xt[k, i] * invlen
        end
        for r in 1:rank
        	storage[timei + r] -= 2.0 * error * sqrtwi * x[idi + r] * invlen
        end
        for r in 1:rank
        	storage[idi + r] -= 2.0 * error * sqrtwi * x[timei + r] * invlen
        end
    end

    # Tikhonov term
    @inbounds @simd for i in (n_regressors+1):length(x)
        storage[i] += 2.0 * lambda * x[i]
    end
    return storage 
end

# function + gradient in the same loop
function sum_of_squares_and_gradient!{Tid, Ttime}(x::Vector{Float64}, storage::Vector{Float64}, sqrtw::AbstractVector{Float64}, y::Vector{Float64}, timerefs::Vector{Ttime}, idrefs::Vector{Tid}, n_regressors::Integer, rank::Integer, Xt::Matrix{Float64}, lambda::Real, invlen::Real)
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
			storage[k] -= 2.0 * error  * Xt[k, i] * invlen
		end
		for r in 1:rank
			storage[timei + r] -= 2.0 * error * sqrtwi * x[idi + r] * invlen
		end
		for r in 1:rank
			storage[idi + r] -= 2.0 * error * sqrtwi * x[timei + r] * invlen
		end
    end
    out *= invlen

    # Tikhonov term
    @inbounds @simd for i in (n_regressors+1):length(x)
        out += lambda * abs2(x[i])
        storage[i] += 2.0 * lambda * x[i]
    end
    return out 
end



##############################################################################
##
## Estimate linear factor model by original Bai (2009) method: SVD / beta iteration
##
##############################################################################


function fit_svd{Tid, Rid, Ttime, Rtime}(X::Matrix{Float64}, M::Matrix{Float64}, b::Vector{Float64}, y::Vector{Float64}, id::PooledDataVector{Tid, Rid}, time::PooledDataVector{Ttime, Rtime}, rank::Integer ; maxiter::Integer = 100_000, tol::Real = 1e-9)
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
	newfactors = reverse(factors)
	loadings = predict_matrix * newfactors
	return (b, loadings, newfactors, [iterations], [converged])
end

