##############################################################################
##
## Estimate linear factor model by gradient method
##
##############################################################################

function fit_gd!{Rid, Rtime}(X::Matrix{Float64}, M::Matrix{Float64}, b::Vector{Float64}, y::Vector{Float64}, idf::PooledFactor{Rid}, timef::PooledFactor{Rtime}, sqrtw::AbstractVector{Float64}; maxiter::Integer = 100_000, tol::Real = 1e-9, lambda::Real = 0.0)

	rank = size(idf.pool, 2)
	N = size(idf.pool, 1)
	T = size(timef.pool, 1)

	res = deepcopy(y)
	new_b = deepcopy(b)


	# starts loop
	converged = false
	iterations = maxiter
	iter = 0
	learning_rate = fill(0.1, rank)

	copy!(idf.old1pool, idf.pool)
	copy!(timef.old1pool, timef.pool)
	copy!(idf.old2pool, idf.pool)
	copy!(timef.old2pool, timef.pool)


	Xt = X'
	error = Inf
	olderror = Inf
	while iter < maxiter
		iter += 1
		(error, olderror) = (olderror, error)

		# Given beta, compute incrementally an approximate factor model
		subtract_b!(res, y, b, X)
		for r in 1:rank
			steps_in_a_row = 0
			olderror_inner = ssr(idf, timef, res, sqrtw, r)
			# recompute error since res has changed in the meantime
			while steps_in_a_row <= 4
				update_gd!(idf, timef, res, sqrtw, r, learning_rate[r], lambda)
				error_inner = ssr(idf, timef, res, sqrtw, r)
				if error_inner < olderror_inner
					olderror_inner = error_inner
				    steps_in_a_row = max(1, steps_in_a_row + 1)
				    # increase learning rate
				    learning_rate[r] *= 1.1
				   # update old2pool
				   (idf.old1pool, idf.old2pool) = (idf.old2pool, idf.old1pool)
				   (timef.old1pool, timef.old2pool) = (timef.old2pool, timef.old1pool)
				   # update old1pool
				   copy!(idf.old1pool, idf.pool)
				   copy!(timef.old1pool, timef.pool)
				else
					# decrease learning rate
				    learning_rate[r] /= max(1.5, -steps_in_a_row)
				    steps_in_a_row = min(0, steps_in_a_row - 1)
				    # cancel the update
				    copy!(idf.pool, idf.old1pool, r)
				    copy!(timef.pool, timef.old1pool, r)
				end		
			end
			# don't rescale since screw up learning_rate
			subtract_factor!(res, sqrtw, idf, timef, r)
		end

		# Given factor model, compute beta
		copy!(res, y)
		subtract_factor!(res, sqrtw, idf, timef)
		b = M * res

		# Check convergence
		error = ssr(idf, timef, b, Xt, y, sqrtw) 
		if error == zero(Float64) || abs(error - olderror)/error < tol 
			converged = true
			iterations = iter
			break
		end
	end

	rescale!(idf.old1pool, timef.old1pool, idf.pool, timef.pool)
	(idf.old1pool, idf.pool) = (idf.pool, idf.old1pool)
	(timef.old1pool, timef.pool) = (timef.pool, timef.old1pool)
	return (b, [iterations], [converged])
end

##############################################################################
##
## Estimate linear factor model by gauss-seidel method
##
##############################################################################

function fit_ar!{Rid, Rtime}(X::Matrix{Float64}, M::Matrix{Float64}, b::Vector{Float64}, y::Vector{Float64}, idf::PooledFactor{Rid}, timef::PooledFactor{Rtime}, sqrtw::AbstractVector{Float64}; maxiter::Integer = 100_000, tol::Real = 1e-9)

	rank = size(idf.pool, 2)
	N = size(idf.pool, 1)
	T = size(timef.pool, 1)

	res = deepcopy(y)
	new_b = deepcopy(b)

	# starts loop
	converged = false
	iterations = maxiter
	iter = 0
	Xt = X'
	error = Inf
	olderror = Inf
	while iter < maxiter
		iter += 1
		(error, olderror) = (olderror, error)

		if mod(iter, 100) == 0
			rescale!(idf.old1pool, timef.old1pool, idf.pool, timef.pool)
			copy!(idf.pool, idf.old1pool)
			copy!(timef.pool, timef.old1pool)
		end
		
		# Given beta, compute incrementally an approximate factor model
		subtract_b!(res, y, b, X)
		for r in 1:rank
			update_ar!(idf, timef, res, sqrtw, r)
			subtract_factor!(res, sqrtw, idf, timef, r)
		end

		# Given factor model, compute beta
		copy!(res, y)
		subtract_factor!(res, sqrtw, idf, timef)
		b = M * res

		# Check convergence
		error = ssr(idf, timef, b, Xt, y, sqrtw) 
		if error == zero(Float64) || abs(error - olderror)/error < tol 
			converged = true
			iterations = iter
			break
		end
	end

	rescale!(idf.old1pool, timef.old1pool, idf.pool, timef.pool)
	(idf.old1pool, idf.pool) = (idf.pool, idf.old1pool)
	(timef.old1pool, timef.pool) = (timef.pool, timef.old1pool)
	return (b, [iterations], [converged])
end


##############################################################################
##
## Estimate linear factor model by SVD method
##
##############################################################################


function fit_svd!{Rid, Rtime}(X::Matrix{Float64}, M::Matrix{Float64}, b::Vector{Float64}, y::Vector{Float64}, idf::PooledFactor{Rid}, timef::PooledFactor{Rtime} ; maxiter::Integer = 100_000, tol::Real = 1e-9)
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
	error = Inf
	olderror = Inf

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
		error = chebyshev(b, new_b)
		if error < tol 
			converged = true
			iterations = iter
			break
		end
	end

	timef.pool = reverse(factors)
	idf.pool = predict_matrix * timef.pool
	return (b, [iterations], [converged])
end


