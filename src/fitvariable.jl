
##############################################################################
##
## Estimate factor model by incremental optimization routine
##
##############################################################################

function fit_optimization{Tid, Rid, Ttime, Rtime}(y::Vector{Float64}, id::PooledDataVector{Tid, Rid}, time::PooledDataVector{Ttime, Rtime}, rank::Integer, sqrtw::AbstractVector{Float64}; lambda::Real = 0.0,  method::Symbol = :gradient_descent, maxiter::Integer = 100_000, tol::Real = 1e-9)
    
    invlen = 1 / abs2(norm(sqrtw, 2)) 

    # initialize
    iterations = fill(maxiter, rank)
    converged = fill(false, rank)
    loadings = Array(Float64, (length(id.pool), rank))
    factors = Array(Float64, (length(time.pool), rank))

    # squeeze (loadings and factors) -> x0
    l = length(id.pool)
    x0 = fill(0.1, length(id.pool) + length(time.pool))
    res = deepcopy(y)
    for r in 1:rank
        # set up optimization problem
        f = x -> sum_of_squares(x, sqrtw, res, time.refs, id.refs, l, lambda, invlen)
        g! = (x, storage) -> sum_of_squares_gradient!(x, storage, sqrtw, res, time.refs, id.refs, l, lambda, invlen)
        fg! = (x, storage) -> sum_of_squares_and_gradient!(x, storage, sqrtw, res, time.refs, id.refs, l, lambda, invlen)
        d = DifferentiableFunction(f, g!, fg!)

        # optimize
        # xtol corresponds to maxdiff(x, x_previous)
        result = optimize(d, x0, method = method, iterations = maxiter, xtol = -nextfloat(0.0), ftol = tol, grtol = -nextfloat(0.0))
        
        # develop minimumm -> (loadings and factors)
        loadings[:, r] = result.minimum[1:l]
        factors[:, r] = result.minimum[(l+1):end]
        iterations[r] = result.iterations
        converged[r] = result.x_converged || result.f_converged || result.gr_converged
        
        # take the residuals res - lambda_i * ft
        subtract_factor!(res, sqrtw, id.refs, loadings, time.refs, factors, r)
    end
    (newloadings, newfactors) = rescale(loadings, factors)

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
    out *= invlen

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
        storage[idi] -= 2.0 * error * sqrtwi * factor  * invlen
        storage[timei] -= 2.0 * error * sqrtwi * loading * invlen
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
        storage[idi] -= 2.0 * error * sqrtwi * factor * invlen
        storage[timei] -= 2.0 * error * sqrtwi * loading * invlen
    end
    
    out *= invlen
    # Tikhonov term
    @inbounds @simd for i in 1:length(x)
        out += lambda * abs2(x[i])
        storage[i] += 2.0 * lambda * x[i]
    end

    return out
end


##############################################################################
##
## Estimate factor model by EM Method
##
##############################################################################

function fit_svd{Tid, Rid, Ttime, Rtime}(y::Vector{Float64}, ids::PooledDataVector{Tid, Rid}, times::PooledDataVector{Ttime, Rtime}, rank::Integer; maxiter::Integer = 100_000, tol::Real = 1e-8)
 
    # initialize at zero for missing values
    res_matrix = fill(zero(Float64), (length(ids.pool), length(times.pool)))
    predict_matrix = deepcopy(res_matrix)
    factors = Array(Float64, (length(times.pool), rank))
    variance = Array(Float64, (length(times.pool), length(times.pool)))
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
        fill!(res_matrix, y, ids.refs, times.refs)

        # principal components
        At_mul_B!(variance, res_matrix, res_matrix)
        F = eigfact!(Symmetric(variance), (length(times.pool) - rank + 1):length(times.pool))
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
## Estimate factor model by incremental backpropagation (Simon Funk Netflix Algorithm)
##
##############################################################################

function fit_backpropagation{Tids, Rids, Ttimes, Rtimes}(y::Vector{Float64}, id::PooledDataVector{Tids, Rids}, time::PooledDataVector{Ttimes, Rtimes}, rank::Integer, sqrtw::AbstractVector{Float64} ; regularizer::Real = 0.0, learning_rate::Real = 1e-3, maxiter::Integer = 100_000, tol::Real = 1e-9)
    
    # initialize
    idf = PooledFactor(id.refs, length(id.pool), rank)
    timef = PooledFactor(time.refs, length(time.pool), rank)
    iterations = fill(maxiter, rank)
    converged = fill(false, rank)

    res = deepcopy(y)
    for r in 1:rank
    	error = zero(Float64)
        olderror = zero(Float64)
        iter = 0
        while iter < maxiter
            iter += 1
            (error, olderror) = (olderror, error)
           	error = update!(idf, timef, res, sqrtw, regularizer, learning_rate / iter, r)
            # relative tolerance (absolute change would depend on learning_rate choice)
            if error == zero(Float64) || abs(error - olderror)/error < tol 
                iterations[r] = iter
                converged[r] = true
                break
            end
        end
        rescale!(idf, timef, r)
        subtract_factor!(res, sqrtw, idf, timef, r)
    end
    (loadings, factors) = rescale(idf.pool, timef.pool)
    return (loadings, factors, iterations, converged)
end


##############################################################################
##
## Estimate factor model by gs algorithm
##
##############################################################################

function fit_gs{Tid, Rid, Ttime, Rtime}(y::Vector{Float64}, id::PooledDataVector{Tid, Rid}, time::PooledDataVector{Ttime, Rtime}, rank::Integer, sqrtw::AbstractVector{Float64}; maxiter::Integer  = 100_000, tol::Real = 1e-9)

    # initialize
    idf = PooledFactor(id.refs, length(id.pool), rank)
    timef = PooledFactor(time.refs, length(time.pool), rank)
    iterations = fill(maxiter, rank)
    converged = fill(false, rank)

    iter = 0
    res = deepcopy(y)
    for r in 1:rank
        error = zero(Float64)
        olderror = zero(Float64)
        iter = 0
        while iter < maxiter
            iter += 1
            error = update!(idf, timef, res, sqrtw, r)
            rescale!(idf, timef, r)
            # relative tolerance (absolute change would depend on learning_rate choice)
            if error == zero(Float64) || abs(error - olderror)/error < tol 
                iterations[r] = iter
                converged[r] = true
                break
            end
        end
        rescale!(idf, timef, r)
        subtract_factor!(res, sqrtw, idf, timef, r)
    end
    (loadings, factors) = rescale(idf.pool, timef.pool)

    return (loadings, factors, [iterations], [converged])
end

