##############################################################################
##
## Estimate factor model by gradient method
##
##############################################################################

function fit_gr{Rid, Rtime}(y::Vector{Float64}, idf::PooledFactor{Rid}, timef::PooledFactor{Rtime}, sqrtw::AbstractVector{Float64}; maxiter::Integer  = 100_000, tol::Real = 1e-9, alpha = 0.5)

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
        learning_rate = 1.0
        iter = 0
        steps_in_a_row  = 0
        while iter < maxiter
            iter += 1
            update!(idf, timef, res, sqrtw, r, learning_rate, alpha)
            error = ssr(idf, timef, res, sqrtw, r)
            if error < olderror
                if error == zero(Float64) || (abs(error - olderror)/error < tol  && steps_in_a_row > 3)
                    iterations[r] = iter
                    converged[r] = true
                    break
                end
                olderror = error
                steps_in_a_row = max(1, steps_in_a_row + 1)
                learning_rate *= 1.05
                copy!(idf.old2pool, idf.old1pool, r)
                copy!(timef.old2pool, timef.old1pool, r)
                # update
                copy!(idf.old1pool, idf.pool, r)
                copy!(timef.old1pool, timef.pool, r)
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
    return (idf.old1pool, timef.old1pool, [iterations], [converged])
end


##############################################################################
##
## Estimate factor model by GS method
##
##############################################################################

function fit_gs{Rid, Rtime}(y::Vector{Float64}, idf::PooledFactor{Rid}, timef::PooledFactor{Rtime}, sqrtw::AbstractVector{Float64}; maxiter::Integer  = 100_000, tol::Real = 1e-9)

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
            update!(idf, timef, res, sqrtw, r)
            error = ssr(idf, timef, res, sqrtw, r)
            # relative tolerance (absolute change would depend on learning_rate choice)
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

    return (idf.old1pool, timef.old1pool, [iterations], [converged])
end

##############################################################################
##
## Estimate factor model by SVD Method
##
##############################################################################

function fit_svd{Rid, Rtime}(y::Vector{Float64}, idf::PooledFactor{Rid}, timef::PooledFactor{Rtime}; maxiter::Integer = 100_000, tol::Real = 1e-9)
 

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
    newfactors = reverse(factors)
    loadings = res_matrix * newfactors

    return (loadings, newfactors, iterations, converged)

end

##############################################################################
##
## Estimate factor model by incremental optimization routine
##
##############################################################################

function fit_optimization{Rid, Rtime}(y::Vector{Float64}, idf::PooledFactor{Rid}, timef::PooledFactor{Rtime}, sqrtw::AbstractVector{Float64}; lambda::Real = 0.0,  method::Symbol = :gradient_descent, maxiter::Integer = 100_000, tol::Real = 1e-9)
    
    invlen = 1 / abs2(norm(sqrtw, 2)) 
    N = size(idf.pool, 1)
    T = size(timef.pool, 1)
    rank = size(idf.pool, 2)
    # initialize
    iterations = fill(maxiter, rank)
    converged = fill(false, rank)


    # squeeze (loadings and factors) -> x0
    x0 = fill(0.1, N + T)
    res = deepcopy(y)
    for r in 1:rank
        # set up optimization problem
        f = x -> sum_of_squares(x, sqrtw, res, timef.refs, idf.refs, N, lambda, invlen)
        g! = (x, storage) -> sum_of_squares_gradient!(x, storage, sqrtw, res, timef.refs, idf.refs, N, lambda, invlen)
        fg! = (x, storage) -> sum_of_squares_and_gradient!(x, storage, sqrtw, res, timef.refs, idf.refs, N, lambda, invlen)
        d = DifferentiableFunction(f, g!, fg!)

        # optimize
        # xtol corresponds to maxdiff(x, x_previous)
        result = optimize(d, x0, method = method, iterations = maxiter, xtol = -nextfloat(0.0), ftol = tol, grtol = -nextfloat(0.0))
        # develop minimumm -> (loadings and factors)
        idf.pool[:, r] = result.minimum[1:N]
        timef.pool[:, r] = result.minimum[(N+1):end]
        iterations[r] = result.iterations
        converged[r] = result.x_converged || result.f_converged || result.gr_converged
        
        # take the residuals res - lambda_i * ft
        subtract_factor!(res, sqrtw, idf.refs, idf.pool, timef.refs, timef.pool, r)
    end
    (newloadings, newfactors) = rescale(idf.pool, timef.pool)

    return (newloadings, newfactors, iterations, converged)
end

# fitness
function sum_of_squares{Ttime, Tid}(x::Vector{Float64}, sqrtw::AbstractVector{Float64}, y::Vector{Float64}, timerefs::Vector{Ttime}, idrefs::Vector{Tid}, l::Integer, lambda::Real, invlen::Real)
    out = zero(Float64)
    @inbounds @simd for i in 1:length(y)
        idi = idrefs[i]
        timei = timerefs[i] + l
        loading = x[idi]
        factor = x[timei]
        sqrtwi = sqrtw[i]
        error = y[i] - sqrtwi * loading * factor
        out += abs2(error)
    end

    # Tikhonov term
    @inbounds @simd for i in 1:length(x)
        out += lambda * abs2(x[i])
    end
    return out 
end

# gradient
function sum_of_squares_gradient!{Ttime, Tid}(x::Vector{Float64}, storage::Vector{Float64}, sqrtw::AbstractVector{Float64}, y::Vector{Float64}, timerefs::Vector{Ttime}, idrefs::Vector{Tid}, l::Integer, lambda::Real, invlen::Real)
    fill!(storage, zero(Float64))
    @inbounds @simd for i in 1:length(y)
        idi = idrefs[i]
        timei = timerefs[i] + l
        loading = x[idi]
        factor = x[timei]
        sqrtwi = sqrtw[i]
        error = y[i] - sqrtwi * loading * factor
        storage[idi] -= 2.0 * error * sqrtwi * factor  
        storage[timei] -= 2.0 * error * sqrtwi * loading 
    end
    
    # Tikhonov term
    @inbounds @simd for i in 1:length(x)
        storage[i] += 2.0 * lambda * x[i]
    end
    return storage
end

# fitness + gradient in the same loop
function sum_of_squares_and_gradient!{Ttime, Tid}(x::Vector{Float64}, storage::Vector{Float64}, sqrtw::AbstractVector{Float64}, y::Vector{Float64}, timerefs::Vector{Ttime}, idrefs::Vector{Tid}, l::Integer, lambda::Real, invlen::Real)
    fill!(storage, zero(Float64))
    len_y = length(y)
    out = zero(Float64)
    @inbounds @simd for i in 1:length(y)
        idi = idrefs[i]
        timei = timerefs[i]+l
        loading = x[idi]
        factor = x[timei]
        sqrtwi = sqrtw[i]
        error =  y[i] - sqrtwi * loading * factor
        out += abs2(error)
        storage[idi] -= 2.0 * error * sqrtwi * factor 
        storage[timei] -= 2.0 * error * sqrtwi * loading 
    end
    

    # Tikhonov term
    @inbounds @simd for i in 1:length(x)
        out += lambda * abs2(x[i])
        storage[i] += 2.0 * lambda * x[i]
    end

    return out
end

