
##############################################################################
##
## Estimate factor model by incremental optimization routine
##
##############################################################################
# fitness
function optim_f{Ttime, Tid}(x::Vector{Float64}, sqrtw::AbstractVector{Float64}, y::Vector{Float64}, timerefs::Vector{Ttime}, idrefs::Vector{Tid}, l::Integer, lambda::Real, invlen::Real)
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

# fitness + gradient 
function optim_fg!{Ttime, Tid}(x::Vector{Float64}, out::Vector{Float64}, sqrtw::AbstractVector{Float64}, y::Vector{Float64}, timerefs::Vector{Ttime}, idrefs::Vector{Tid}, l::Integer, lambda::Real, invlen::Real)
    fill!(out, zero(Float64))
    len_y = length(y)
    sum = zero(Float64)
    @inbounds @simd for i in 1:length(y)
        idi = idrefs[i]
        timei = timerefs[i]+l
        loading = x[idi]
        factor = x[timei]
        sqrtwi = sqrtw[i]
        error =  y[i] - sqrtwi * loading * factor
        sum += abs2(error)
        out[idi] -= 2.0 * error * sqrtwi * factor 
        out[timei] -= 2.0 * error * sqrtwi * loading 
    end
    # Tikhonov term
    @inbounds @simd for i in 1:length(x)
        sum += lambda * abs2(x[i])
        out[i] += 2.0 * lambda * x[i]
    end
    return sum
end



function fit!{R, Rid, Rtime}(::Type{Val{R}},
                        y::Vector{Float64},
                        idf::PooledFactor{Rid},
                        timef::PooledFactor{Rtime},
                        sqrtw::AbstractVector{Float64}; 
                        maxiter::Integer = 100_000,
                        tol::Real = 1e-9,
                        lambda::Real = 0.0)
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
        d = DifferentiableFunction(
            x -> optim_f(x, sqrtw, res, timef.refs, idf.refs, N, lambda, invlen),
            (x, out) -> optim_fg!(x, out, sqrtw, res, timef.refs, idf.refs, N, lambda, invlen),
            (x, out) -> optim_fg!(x, out, sqrtw, res, timef.refs, idf.refs, N, lambda, invlen)
            )

        # optimize
        # xtol corresponds to maxdiff(x, x_previous)
        result = optimize(d, x0, method = R, iterations = maxiter, xtol = -nextfloat(0.0), ftol = tol, grtol = -nextfloat(0.0))
        # develop minimumm -> (loadings and factors)
        idf.pool[:, r] = result.minimum[1:N]
        timef.pool[:, r] = result.minimum[(N+1):end]
        iterations[r] = result.iterations
        converged[r] = result.x_converged || result.f_converged || result.gr_converged
        
        # take the residuals res - lambda_i * ft
        subtract_factor!(res, sqrtw, idf, timef, r)
    end
    rescale!(idf.old1pool, timef.old1pool, idf.pool, timef.pool)
    (idf.old1pool, idf.pool) = (idf.pool, idf.old1pool)
    (timef.old1pool, timef.pool) = (timef.pool, timef.old1pool)
    return (iterations, converged)
end


##############################################################################
##
## Estimate interactive factor model by incremental optimization routine
##
##############################################################################
# fitness
function optim_f{Tid, Ttime}(x::Vector{Float64}, sqrtw::AbstractVector{Float64}, y::Vector{Float64}, timerefs::Vector{Ttime}, idrefs::Vector{Tid}, n_regressors::Integer, rank::Integer, Xt::Matrix{Float64}, lambda::Real, invlen::Real)
    out = zero(Float64)
    # residual term
    @simd for i in 1:length(y)
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
    @simd for i in (n_regressors+1):length(x)
        out += lambda * abs2(x[i])
    end
    return out
end

# fitness + gradient in the same loop
function optim_fg!{Tid, Ttime}(x::Vector{Float64}, out::Vector{Float64}, sqrtw::AbstractVector{Float64}, y::Vector{Float64}, timerefs::Vector{Ttime}, idrefs::Vector{Tid}, n_regressors::Integer, rank::Integer, Xt::Matrix{Float64}, lambda::Real, invlen::Real)
    fill!(out, zero(Float64))
    sum = zero(Float64)
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
        sum += abs2(error)
        for k in 1:n_regressors
            out[k] -= 2.0 * error  * Xt[k, i] * invlen
        end
        for r in 1:rank
            out[timei + r] -= 2.0 * error * sqrtwi * x[idi + r] * invlen
        end
        for r in 1:rank
            out[idi + r] -= 2.0 * error * sqrtwi * x[timei + r] * invlen
        end
    end
    sum *= invlen

    # Tikhonov term
    @inbounds @simd for i in (n_regressors+1):length(x)
        sum += lambda * abs2(x[i])
        out[i] += 2.0 * lambda * x[i]
    end
    return sum 
end



function fit!{R, Rid, Rtime}(::Type{Val{R}}, 
                         X::Matrix{Float64},
                         M::Matrix{Float64},
                         b::Vector{Float64},
                         y::Vector{Float64},
                         idf::PooledFactor{Rid},
                         timef::PooledFactor{Rtime},
                         sqrtw::AbstractVector{Float64}; 
                         maxiter::Integer = 100_000,
                         tol::Real = 1e-9,
                         lambda::Real = 0.0)

    n_regressors = size(X, 2)
    invlen = 1 / abs2(norm(sqrtw, 2)) 
    rank = size(idf.pool, 2)
    res = deepcopy(y)
    N = size(idf.pool, 1)
    T = size(timef.pool, 1)

    # squeeze (b, loadings and factors) into a vector x0
    x0 = Array(Float64, n_regressors + rank * N + rank * T)
    x0[1:n_regressors] = b

    idx = n_regressors
    @inbounds for i in 1:size(idf.pool, 1)
        for r in 1:rank
            idx += 1
            x0[idx] = idf.pool[i, r]
        end
    end
    @inbounds for i in 1:size(timef.pool, 1)
      for r in 1:rank
          idx += 1
          x0[idx] = timef.pool[i, r]
      end
    end

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
    d = DifferentiableFunction(
        x -> optim_f(x, sqrtw, y, timerefs, idrefs, n_regressors, rank, Xt, lambda, invlen),
        (x, out) ->  optim_fg!(x, out, sqrtw, y, timerefs, idrefs, n_regressors, rank, Xt, lambda, invlen),
        (x, out) ->  optim_fg!(x, out, sqrtw, y, timerefs, idrefs, n_regressors, rank, Xt, lambda, invlen)
        )
    # convergence is chebyshev for x
    result = optimize(d, x0, method = R, iterations = maxiter, xtol = tol, ftol = -nextfloat(0.0), grtol = -nextfloat(0.0))
    minimizer = result.minimum
    iterations = result.iterations
    converged =  result.x_converged || result.f_converged || result.gr_converged

    # expand minimumm -> (b, loadings and factors)
    b = minimizer[1:n_regressors]
    idx = n_regressors
   @inbounds for i in 1:size(idf.pool, 1)
          for r in 1:rank
              idx += 1
              idf.old1pool[i, r] = minimizer[idx] 
          end
      end
      @inbounds for i in 1:size(timef.pool, 1)
        for r in 1:rank
            idx += 1
            timef.old1pool[i, r] = minimizer[idx] 
        end
      end

    # rescale factors and loadings so that factors' * factors = Id
    rescale!(idf.pool, timef.pool, idf.old1pool, timef.old1pool)
    return (b, [iterations], [converged])
end





