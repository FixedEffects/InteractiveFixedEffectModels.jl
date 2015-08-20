##############################################################################
##
## update! by stochastic gradient
##
##############################################################################

function update!{R1, R2}(::Type{Val{:gsd}}, 
                         id::PooledFactor{R1}, 
                         time::PooledFactor{R2}, 
                         y::Vector{Float64}, 
                         sqrtw::AbstractVector{Float64}, 
                         r::Int, 
                         learning_rate::Real, 
                         lambda::Real, 
                         range::UnitRange{Int})
     @inbounds for j in 1:length(y)
        i = rand(range)
        idi = id.refs[i]
        timei = time.refs[i]
        loading = id.pool[idi, r]
        factor = time.pool[timei, r]
        sqrtwi = sqrtw[i]
        error = y[i] - sqrtwi * loading * factor 
        id.pool[idi, r] += learning_rate * 2.0 * (error * sqrtwi * factor - lambda * loading)
        time.pool[timei, r] += learning_rate * 2.0 * (error * sqrtwi * loading - lambda * factor)
    end   
end

##############################################################################
##
## Estimate factor model by stochastic gradient method
## It seems there is some issue because thinks converged but did not
## 
##############################################################################

function fit!{Rid, Rtime}(::Type{Val{:gsd}},
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
    res = deepcopy(y)
    copy!(idf.old1pool, idf.pool)
    copy!(timef.old1pool, timef.pool)
    copy!(idf.old2pool, idf.pool)
    copy!(timef.old2pool, timef.pool)
    range = 1:length(y)
    history = Float64[]

    iter = 0
    for r in 1:rank
        olderror = ssr(idf, timef, y, sqrtw, r) + ssr_penalty(idf, timef, lambda, r)
        learning_rate = 0.01
        iter = 0
        steps_in_a_row  = 0

        while iter < maxiter
            iter += 1
            update!(Val{:sgd}, idf, timef, res, sqrtw, r, learning_rate, lambda, range)
            error = ssr(idf, timef, res, sqrtw, r) + ssr_penalty(idf, timef, lambda, r)
            push!(history, error)
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
        # don't rescale during algorithm due to learning rate
        if r < rank
            rescale!(idf, timef, r)
            subtract_factor!(res, sqrtw, idf, timef, r)
        end
    end

    rescale!(idf.old1pool, timef.old1pool, idf.pool, timef.pool)
    (idf.old1pool, idf.pool) = (idf.pool, idf.old1pool)
    (timef.old1pool, timef.pool) = (timef.pool, timef.old1pool)
    @show history
    return (iterations, converged)
end

