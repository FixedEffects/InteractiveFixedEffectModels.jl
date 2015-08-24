##############################################################################
##
## update! by gradient method
##
##############################################################################

function gd_f{R1, R2}(x::Vector{Float64}, p1::PooledFactor{R1}, p2::PooledFactor{R2},  y::Vector{Float64}, sqrtw::AbstractVector{Float64}, r::Integer)
    f_x = zero(Float64)
    @inbounds @simd for i in 1:length(y)
        idi = p1.refs[i]
        timei = p2.refs[i]
        loading = x[idi]
        factor = p2.pool[timei, r]
        sqrtwi = sqrtw[i]
        error = y[i] - sqrtwi * loading * factor 
        f_x += error^2
    end
    return f_x
end

function gd_g!{R1, R2}(x::Vector{Float64}, p1::PooledFactor{R1}, p2::PooledFactor{R2},  y::Vector{Float64}, sqrtw::AbstractVector{Float64}, r::Integer)
    fill!(out, zero(Float64))
    @inbounds @simd for i in 1:length(y)
        idi = p1.refs[i]
        timei = p2.refs[i]
        loading = x[idi]
        factor = p2.pool[timei, r]
        sqrtwi = sqrtw[i]
        error = y[i] - sqrtwi * loading * factor 
        out[idi] -= 2.0 * error * sqrtwi * factor
    end
end


function gd_fg!{R1, R2}(x::Vector{Float64}, out::Vector{Float64}, p1::PooledFactor{R1}, p2::PooledFactor{R2},  y::Vector{Float64}, sqrtw::AbstractVector{Float64}, r::Integer)
    f_x = zero(Float64)
    fill!(out, zero(Float64))
    @inbounds @simd for i in 1:length(y)
        idi = p1.refs[i]
        timei = p2.refs[i]
        loading = x[idi]
        factor = p2.pool[timei, r]
        sqrtwi = sqrtw[i]
        error = y[i] - sqrtwi * loading * factor 
        out[idi] -= 2.0 * error * sqrtwi * factor
        f_x += error^2
    end
    return f_x
end

function update!{R1, R2}(::Type{Val{:gd}}, 
                         id::PooledFactor{R1},
                         time::PooledFactor{R2},
                         y::Vector{Float64},
                         sqrtw::AbstractVector{Float64},
                         r::Integer,
                         learning_rate::Vector{Float64},
                         lambda::Float64)
    learning_rate[1] = update_half!(Val{:gd}, id, time, y, sqrtw, r, learning_rate[1], lambda)
    learning_rate[2] = update_half!(Val{:gd}, time, id, y, sqrtw, r, learning_rate[2], lambda)
end

function update_half!{R1, R2}(::Type{Val{:gd}},
                              p1::PooledFactor{R1},
                              p2::PooledFactor{R2},
                              y::Vector{Float64},
                              sqrtw::AbstractVector{Float64},
                              r::Integer, 
                              learning_rate::Float64, 
                              lambda::Float64)


    # construct differntiable function for use with Optim package
    d = Optim.DifferentiableFunction(
        x -> gd_f(x, p1, p2,  y, sqrtw, r), 
        (x, out) -> gd_g!(x, out, p1, p2,  y, sqrtw, r),
        (x, out) -> gd_fg!(x, out, p1, p2,  y, sqrtw, r)
    )

    # update p1.x
    copy!(p1.x, slice(p1.pool, :, r))
    f_x = d.fg!(p1.x, p1.gr)
    dphi0 = -sumabs2(p1.gr)

    # build lsr
    lsr = Optim.LineSearchResults(Float64)
    Optim.clear!(lsr)
    Optim.push!(lsr, zero(Float64), f_x, dphi0)

    # direction is minus gradient
    scale!(p1.gr, -1.0)

    # linesearch
    learning_rate, f_update, g_update =
              Optim.hz_linesearch!(d, p1.x, p1.gr, p1.x_ls, p1.gr_ls, lsr, learning_rate, false)

    # update
    for i in 1:length(p1.x)
        p1.pool[i, r] =  p1.pool[i, r] + learning_rate * p1.gr[i]
    end

    return learning_rate
end


##############################################################################
##
## Estimate factor model by gradient descent method
##
##############################################################################

function fit!{Rid, Rtime}(::Type{Val{:gd}},
                          y::Vector{Float64}, 
                          idf::PooledFactor{Rid}, 
                          timef::PooledFactor{Rtime}, 
                          sqrtw::AbstractVector{Float64}; 
                          maxiter::Integer  = 100_000, 
                          tol::Real = 1e-9, 
                          lambda::Real = 0.0)

    # initialize
    rank = size(idf.pool, 2)
    iterations = fill(maxiter, rank)
    converged = fill(false, rank)
    history = Float64[]

    iter = 0
    res = deepcopy(y)
    copy!(idf.old1pool, idf.pool)
    copy!(timef.old1pool, timef.pool)
    copy!(idf.old2pool, idf.pool)
    copy!(timef.old2pool, timef.pool)

    for r in 1:rank
        learning_rate = fill(1.0, 2)
        iter = 0
        steps_in_a_row  = 0
        oldf_x = Inf
        while iter < maxiter
            iter += 1
            update!(Val{:gd}, idf, timef, res, sqrtw, r, learning_rate, lambda)
            error = ssr(idf, timef, res, sqrtw, r) + ssr_penalty(idf, timef, lambda, r)
            push!(history, error)
            if error == zero(Float64) || abs(error - oldf_x)/error < tol  
                iterations[r] = iter
                converged[r] = true
                break
            end
            oldf_x = error
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
    return (iterations, converged)
end


##############################################################################
##
## Estimate ols models with interactive fixed effects by gradient descent
##
##############################################################################

function fit!{Rid, Rtime}(::Type{Val{:gd}},
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

    rank = size(idf.pool, 2)
    N = size(idf.pool, 1)
    T = size(timef.pool, 1)

    res = deepcopy(y)
    new_b = deepcopy(b)


    # starts loop
    converged = false
    iterations = maxiter
    iter = 0
    learning_rate = Array(Vector{Float64}, rank)
    for r in 1:rank
        learning_rate[r] = fill(1.0, 2)
    end

    copy!(idf.old1pool, idf.pool)
    copy!(timef.old1pool, timef.pool)
    copy!(idf.old2pool, idf.pool)
    copy!(timef.old2pool, timef.pool)

    Xt = X'
    f_x = Inf
    oldf_x = Inf
    while iter < maxiter
        iter += 1
        (f_x, oldf_x) = (oldf_x, f_x)

        # Given beta, compute incrementally an approximate factor model
        copy!(res, y)
        subtract_b!(res, b, X)
        for r in 1:rank
            update!(Val{:gd}, idf, timef, res, sqrtw, r, learning_rate[r], lambda)
            subtract_factor!(res, sqrtw, idf, timef, r)
        end

        # Given factor model, compute beta
        copy!(res, y)
        subtract_factor!(res, sqrtw, idf, timef)
        b = M * res 

        # Check convergence
        subtract_b!(res, b, X)
        f_x = sumabs2(res) + ssr_penalty(idf, timef, lambda)
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