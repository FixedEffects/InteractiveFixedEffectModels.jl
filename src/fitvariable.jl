##############################################################################
##
## Estimate factor model by alternative regression
##
##############################################################################

function fit_ar!{Rid, Rtime}(y::Vector{Float64}, idf::PooledFactor{Rid}, timef::PooledFactor{Rtime}, sqrtw::AbstractVector{Float64}; maxiter::Integer  = 100_000, tol::Real = 1e-9)

    # initialize
    rank = size(idf.pool, 2)
    iterations = fill(maxiter, rank)
    converged = fill(false, rank)

    iter = 0
    res = deepcopy(y)
    for r in 1:rank
        olderror = ssr(idf, timef, res, sqrtw, r)
        iter = 0
        while iter < maxiter
            iter += 1
            update_ar!(idf, timef, res, sqrtw, r)
            error = ssr(idf, timef, res, sqrtw, r)
            if error == zero(Float64) || abs(error - olderror)/error < tol 
                iterations[r] = iter
                converged[r] = true
                break
            else
                olderror = error
            end
        end
        if r < rank
            rescale!(idf, timef, r)
            subtract_factor!(res, sqrtw, idf, timef, r)
        end
    end
    rescale!(idf.old1pool, timef.old1pool, idf.pool, timef.pool)
    (idf.old1pool, idf.pool) = (idf.pool, idf.old1pool)
    (timef.old1pool, timef.pool) = (timef.pool, timef.old1pool)

    return ([iterations], [converged])
end

##############################################################################
##
## Estimate factor model by gradient descent method
##
##############################################################################

function fit_gd!{Rid, Rtime}(y::Vector{Float64}, idf::PooledFactor{Rid}, timef::PooledFactor{Rtime}, sqrtw::AbstractVector{Float64}; maxiter::Integer  = 100_000, tol::Real = 1e-9, lambda::Real = 0.0)

    # initialize
    rank = size(idf.pool, 2)
    iterations = fill(maxiter, rank)
    converged = fill(false, rank)

    iter = 0
    res = deepcopy(y)
    copy!(idf.old1pool, idf.pool)
    copy!(timef.old1pool, timef.pool)
    copy!(idf.old2pool, idf.pool)
    copy!(timef.old2pool, timef.pool)

    for r in 1:rank
        olderror = ssr(idf, timef, y, sqrtw, r)
        learning_rate = 0.1
        iter = 0
        steps_in_a_row  = 0
        while iter < maxiter
            iter += 1
            update_gd!(idf, timef, res, sqrtw, r, learning_rate, lambda)
            error = ssr(idf, timef, res, sqrtw, r)
            if error < olderror
                if error == zero(Float64) || (abs(error - olderror)/error < tol  && steps_in_a_row > 3)
                    iterations[r] = iter
                    converged[r] = true
                    break
                end
                olderror = error
                steps_in_a_row = max(1, steps_in_a_row + 1)
                learning_rate *= 1.1

                # update old2pool
                (idf.old1pool, idf.old2pool) = (idf.old2pool, idf.old1pool)
                (timef.old1pool, timef.old2pool) = (timef.old2pool, timef.old1pool)
                # update old1pool
                copy!(idf.old1pool, idf.pool)
                copy!(timef.old1pool, timef.pool)
            else
                learning_rate /= max(1.5, -steps_in_a_row)
                steps_in_a_row = min(0, steps_in_a_row - 1)
                copy!(idf.pool, idf.old1pool, r)
                copy!(timef.pool, timef.old1pool, r)
            end

        end
        # don't rescale during algorithm due to learning rate
        if r < rank
            rescale!(idf, timef, r)
            subtract_factor!(res, sqrtw, idf, timef, r)
        end
    end
    rescale!(idf.old1pool, timef.old1pool, idf.pool, timef.pool)
    (idf.old1pool, idf.pool) = (idf.pool, idf.old1pool)
    (timef.old1pool, timef.pool) = (timef.pool, timef.old1pool)
    return ([iterations], [converged])
end

##############################################################################
##
## Estimate factor model by stochastic gradient method
## issue because thinks converged but did not
##############################################################################

function fit_sgd!{Rid, Rtime}(y::Vector{Float64}, idf::PooledFactor{Rid}, timef::PooledFactor{Rtime}, sqrtw::AbstractVector{Float64}; maxiter::Integer  = 100_000, tol::Real = 1e-9, lambda::Real = 0.0)

    # initialize
    rank = size(idf.pool, 2)
    iterations = fill(maxiter, rank)
    converged = fill(false, rank)


    iter = 0
    res = deepcopy(y)
    copy!(idf.old1pool, idf.pool)
    copy!(timef.old1pool, timef.pool)
    copy!(idf.old2pool, idf.pool)
    copy!(timef.old2pool, timef.pool)
    learning_rate = 0.1

    for r in 1:rank
        olderror = ssr(idf, timef, y, sqrtw, r)
        # learning_rate = 1.0
        learning_rate = 0.01
        iter = 0
        steps_in_a_row  = 0

        while iter < maxiter
            iter += 1
            update_sgd!(idf, timef, res, sqrtw, r, learning_rate, lambda)
            error = ssr(idf, timef, res, sqrtw, r)
            if error < olderror
                if error == zero(Float64) || (abs(error - olderror)/error < tol  && steps_in_a_row > 3)
                    iterations[r] = iter
                    converged[r] = true
                    break
                end
                olderror = error
                learning_rate *= 1.1
                steps_in_a_row = max(1, steps_in_a_row + 1)
                 # update old2pool
                 (idf.old1pool, idf.old2pool) = (idf.old2pool, idf.old1pool)
                 (timef.old1pool, timef.old2pool) = (timef.old2pool, timef.old1pool)
                 # update old1pool
                 copy!(idf.old1pool, idf.pool)
                 copy!(timef.old1pool, timef.pool)
            else
                learning_rate /= max(1.5, -steps_in_a_row)
                steps_in_a_row = min(0, steps_in_a_row - 1)
                copy!(idf.pool, idf.old1pool, r)
                copy!(timef.pool, timef.old1pool, r)
            end

        end
        # don't rescal eduring algorithm due to learning rate
        if r < rank
            rescale!(idf, timef, r)
            subtract_factor!(res, sqrtw, idf, timef, r)
        end
    end

    rescale!(idf.old1pool, timef.old1pool, idf.pool, timef.pool)
    (idf.old1pool, idf.pool) = (idf.pool, idf.old1pool)
    (timef.old1pool, timef.pool) = (timef.pool, timef.old1pool)

    return ([iterations], [converged])
end

##############################################################################
##
## Estimate factor model by SVD Method
##
##############################################################################

function fit_svd!{Rid, Rtime}(y::Vector{Float64}, idf::PooledFactor{Rid}, timef::PooledFactor{Rtime}; maxiter::Integer = 100_000, tol::Real = 1e-9)
 

    N = size(idf.pool, 1)
    T = size(timef.pool, 1)
    rank = size(idf.pool, 2)
    # initialize at zero for missing values
    res_matrix = A_mul_Bt(idf.pool, timef.pool)
    predict_matrix = deepcopy(res_matrix)
    factors = timef.pool
    variance = Array(Float64, (T, T))
    converged = Bool[false]
    iterations = Int[maxiter]
    error = zero(Float64)
    olderror = zero(Float64)


    # starts the loop
    iter = 0
    while iter < maxiter
        iter += 1
        (predict_matrix, res_matrix) = (res_matrix, predict_matrix)
        (error, olderror) = (olderror, error)
        # transform vector into matrix
        fill!(res_matrix, y, idf.refs, timef.refs)

        # principal components
        At_mul_B!(variance, res_matrix, res_matrix)
        F = eigfact!(Symmetric(variance), (T - rank + 1):T)
        factors = F[:vectors]
        
        # predict matrix
        A_mul_Bt!(variance, factors, factors)
        A_mul_B!(predict_matrix, res_matrix, variance)

        # check convergence
        error = sqeuclidean(predict_matrix, res_matrix)
        if error == zero(Float64) || abs(error - olderror)/error < tol 
            converged[1] = true
            iterations[1] = iter
            break
        end
    end
    timef.pool = reverse(factors)
    idf.pool = res_matrix * timef.pool

    return (iterations, converged)

end

