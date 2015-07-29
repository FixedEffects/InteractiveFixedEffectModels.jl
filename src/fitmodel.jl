
##############################################################################
##
## Estimate linear factor model by gauss-seidel method
##
##############################################################################

function fit_gs{Rid, Rtime}(X::Matrix{Float64}, M::Matrix{Float64}, b::Vector{Float64}, y::Vector{Float64}, idf::PooledFactor{Rid}, timef::PooledFactor{Rtime}, sqrtw::AbstractVector{Float64}; maxiter::Integer = 100_000, tol::Real = 1e-9)

	rank = size(idf.pool, 2)
	N = size(idf.pool, 1)
	T = size(timef.pool, 1)

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
## Estimate linear factor model by original Bai (2009) method: SVD / beta iteration
##
##############################################################################


function fit_svd{Rid, Rtime}(X::Matrix{Float64}, M::Matrix{Float64}, b::Vector{Float64}, y::Vector{Float64}, idf::PooledFactor{Rid}, timef::PooledFactor{Rtime} ; maxiter::Integer = 100_000, tol::Real = 1e-9)
	b = M * y
	new_b = deepcopy(b)
	res_vector = Array(Float64, length(y))
	N = size(idf.pool, 1)
	T = size(timef.pool, 1)
	rank = size(idf.pool, 2)

	# initialize at zero for missing values
	res_matrix = fill(zero(Float64), (N, T))
	predict_matrix = deepcopy(res_matrix)
	factors = Array(Float64, (T, rank))
	variance = Array(Float64, (T, T))
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
		fill!(res_matrix, res_vector, idf.refs, timef.refs)
		# svd of res_matrix
		At_mul_B!(variance, res_matrix, res_matrix)
		F = eigfact!(Symmetric(variance), (T - rank + 1):T)
		factors = F[:vectors]

		# Given the factors, compute beta
		A_mul_Bt!(variance, factors, factors)
		A_mul_B!(predict_matrix, res_matrix, variance)
		fill!(res_vector, predict_matrix, idf.refs, timef.refs)
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


##############################################################################
##
## Estimate linear factor model by optimization routine
##
##############################################################################

function fit_optimization{Rid, Rtime}(X::Matrix{Float64},  b::Vector{Float64}, y::Vector{Float64}, idf::PooledFactor{Rid}, timef::PooledFactor{Rtime}, sqrtw::AbstractVector{Float64}; method::Symbol = :bfgs, lambda::Real = 0.0, maxiter::Integer = 100_000, tol::Real = 1e-9)

	n_regressors = size(X, 2)
	invlen = 1 / abs2(norm(sqrtw, 2)) 
	rank = size(idf.pool, 2)
	res = deepcopy(y)
	N = size(idf.pool, 1)
	T = size(timef.pool, 1)

	# squeeze (b, loadings and factors) into a vector x0
	x0 = Array(Float64, n_regressors + rank * N + rank * T)
	x0[1:n_regressors] = b
	fill!(x0, idf.pool,  n_regressors)
	fill!(x0, timef.pool,  n_regressors + N * rank)

	# translate indexes
	idrefs = similar(idf.refs)
	@inbounds for i in 1:length(idf.refs)
		idrefs[i] = n_regressors + (idf.refs[i] - 1) * rank 
	end
	timerefs = similar(timef.refs)
	@inbounds for i in 1:length(timef.refs)
		timerefs[i] = n_regressors + N * rank + (timef.refs[i] - 1) * rank 
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
    fill!(idf.pool, minimizer, n_regressors)
    fill!(timef.pool, minimizer, n_regressors + N * rank)  

    # rescale factors and loadings so that factors' * factors = Id
    (scaledloadings, scaledfactors) = rescale(idf.pool, timef.pool)
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
