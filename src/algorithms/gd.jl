##############################################################################
##
## update! by gradient method
##
##############################################################################

function update!{R1, R2}(::Type{Val{:gd}}, 
                         id::PooledFactor{R1},
                         time::PooledFactor{R2},
                         y::Vector{Float64},
                         sqrtw::AbstractVector{Float64},
                         r::Integer,
                         learning_rate::Float64,
                         lambda::Float64)
    update_half!(Val{:gd}, id, time, y, sqrtw, r, learning_rate, lambda)
    update_half!(Val{:gd}, time, id, y, sqrtw, r, learning_rate, lambda)
end

function update_half!{R1, R2}(::Type{Val{:gd}},
                              p1::PooledFactor{R1},
                              p2::PooledFactor{R2},
                              y::Vector{Float64},
                              sqrtw::AbstractVector{Float64},
                              r::Integer, 
                              learning_rate::Float64, 
                              lambda::Float64)
    fill!(p1.storage1, zero(Float64))
    fill!(p1.storage2, zero(Float64))
    @inbounds @simd for i in 1:length(y)
        idi = p1.refs[i]
        timei = p2.refs[i]
        loading = p1.pool[idi, r]
        factor = p2.pool[timei, r]
        sqrtwi = sqrtw[i]
        error = y[i] - sqrtwi * loading * factor 
        p1.storage1[idi] += 2.0 * (error * sqrtwi * factor)
        p1.storage2[idi] += abs2(factor)
    end

    @inbounds @simd for i in 1:size(p1.pool, 1)
        p1.pool[i, r] -= 2.0 * lambda * p1.pool[i, r]
        if p1.storage2[i] > zero(Float64)
            # gradient term is learning rate * p1.storage2
            # newton term (diagonal only) is p1.storage2[i].
            # momentum term. mu = 0 actually makes things faster for factor model
            p1.pool[i, r] += (learning_rate * p1.storage1[i] / p1.storage2[i]^0.5 
                              + 0.00 * (p1.old1pool[i, r] - p1.old2pool[i, r])
                              )
        end
    end
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
        olderror = ssr(idf, timef, y, sqrtw, r) + ssr_penalty(idf, timef, lambda, r)
        learning_rate = 0.1
        iter = 0
        steps_in_a_row  = 0
        while iter < maxiter
            iter += 1
            update!(Val{:gd}, idf, timef, res, sqrtw, r, learning_rate, lambda)
            error = ssr(idf, timef, res, sqrtw, r) + ssr_penalty(idf, timef, lambda, r)
            push!(history, error)
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
    learning_rate = fill(0.1, rank)

    copy!(idf.old1pool, idf.pool)
    copy!(timef.old1pool, timef.pool)
    copy!(idf.old2pool, idf.pool)
    copy!(timef.old2pool, timef.pool)

    Xt = X'
    currenterror = Inf
    olderror = Inf
    while iter < maxiter
        iter += 1
        (currenterror, olderror) = (olderror, currenterror)

        # Given beta, compute incrementally an approximate factor model
        copy!(res, y)
        subtract_b!(res, b, X)
        for r in 1:rank
            steps_in_a_row = 0
            olderror_inner = ssr(idf, timef, res, sqrtw, r) + ssr_penalty(idf, timef, lambda, r)
            # recompute error since res has changed in the meantime
            while steps_in_a_row <= 4 && (learning_rate[r] >= 1e-15 || steps_in_a_row >= 100)
                update!(Val{:gd}, idf, timef, res, sqrtw, r, learning_rate[r], lambda)
                error_inner = ssr(idf, timef, res, sqrtw, r) + ssr_penalty(idf, timef, lambda, r)
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
        subtract_b!(res, b, X)
        currenterror = sumabs2(res) + ssr_penalty(idf, timef, lambda)
        if currenterror == zero(Float64) || abs(currenterror - olderror)/currenterror < tol 
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