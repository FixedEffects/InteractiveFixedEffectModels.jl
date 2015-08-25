##############################################################################
##
## update! by alternative regressions
##
##############################################################################
function update!{R1, R2}(::Type{Val{:ar}},
                         y::Vector{Float64},
                         sqrtw::AbstractVector{Float64},
                         p1::PooledFactor{R1},
                         p2::PooledFactor{R2},
                         r::Integer)
    fill!(p1.x, zero(Float64))
    fill!(p1.x_ls, zero(Float64))
     @inbounds @simd for i in 1:length(p1.refs)
         p1i = p1.refs[i]
         yi = y[i]
         xi = sqrtw[i] * p2.pool[p2.refs[i], r] 
         p1.x[p1i] += xi * yi
         p1.x_ls[p1i] += abs2(xi)
    end
     @inbounds @simd for i in 1:size(p1.pool, 1)
        if p1.x_ls[i] > zero(Float64)
            p1.pool[i, r] = p1.x[i] / p1.x_ls[i]
        end
    end
end

##############################################################################
##
## Estimate factor model by alternative regression
##
##############################################################################

function fit!{Rid, Rtime}(::Type{Val{:ar}}, 
                          y::Vector{Float64}, 
                          idf::PooledFactor{Rid},
                          timef::PooledFactor{Rtime},
                          sqrtw::AbstractVector{Float64};
                          maxiter::Integer  = 100_000,
                          tol::Real = 1e-9,
                          lambda::Real = 0.0
                          )
    lambda == 0.0 || error("The alternative regression method only works with lambda = 0.0")

    # initialize
    rank = size(idf.pool, 2)
    iterations = fill(maxiter, rank)
    converged = fill(false, rank)
    history = Float64[]
    iter = 0
    res = deepcopy(y)
    for r in 1:rank
        oldf_x = ssr(idf, timef, res, sqrtw, r)
        iter = 0
        while iter < maxiter
            iter += 1
            update!(Val{:ar}, res, sqrtw, idf, timef, r)
            update!(Val{:ar}, res, sqrtw, timef, idf, r)
            f_x = ssr(idf, timef, res, sqrtw, r)
            push!(history, f_x)
            if f_x == zero(Float64) || abs(f_x - oldf_x)/f_x < tol 
                iterations[r] = iter
                converged[r] = true
                break
            else
                oldf_x = f_x
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

    return (iterations, converged)
end

##############################################################################
##
## Estimate ols models with interactive fixed effects by alternative regression
##
##############################################################################

function fit!{Rid, Rtime}(::Type{Val{:ar}},
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
    lambda == 0.0 || error("The alternative regression method only works with lambda = 0.0")

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
    f_x = Inf
    oldf_x = Inf
    while iter < maxiter
        iter += 1
        (f_x, oldf_x) = (oldf_x, f_x)

        if mod(iter, 100) == 0
            rescale!(idf.old1pool, timef.old1pool, idf.pool, timef.pool)
            (idf.pool, idf.old1pool) = (idf.old1pool, idf.pool)
            (timef.pool, timef.old1pool) = (timef.old1pool, timef.pool)
        end
        
        # Given beta, compute incrementally an approximate factor model
        copy!(res, y)
        subtract_b!(res, b, X)
        for r in 1:rank
            update!(Val{:ar}, res, sqrtw, timef, idf, r)
            update!(Val{:ar}, res, sqrtw, idf, timef, r)
            subtract_factor!(res, sqrtw, idf, timef, r)
        end

        # Given factor model, compute beta
        copy!(res, y)
        subtract_factor!(res, sqrtw, idf, timef)
        b = M * res

        # Check convergence
        subtract_b!(res, b, X)
        f_x = sumabs2(res)
        if f_x == zero(Float64) || abs(f_x - oldf_x)/f_x < tol 
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