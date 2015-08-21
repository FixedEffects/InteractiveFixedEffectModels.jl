
##############################################################################
##
## Estimate interactive factor model by incremental optimization routine
##
##############################################################################

function fit!{Rid, Rtime}(::Type{Val{:lm2}}, 
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

    @inbounds for i in 1:length(idf.pool)
        x0[n_regressors + i] = idf.pool[i]
    end
    @inbounds for i in 1:length(timef.pool)
        x0[n_regressors + N * rank + i] = timef.pool[i]
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

    # create sparse matrix
    len = length(y)*(n_regressors + 2*rank)
    I = Array(Int, len)
    J = Array(Int, len)
    V = fill(1.0, len)
    idx = zero(Int)
    for i in 1:length(y)
        idi = idrefs[i]
        timei = timerefs[i]
        for k in 1:n_regressors
            idx += 1
            I[idx] = i
            J[idx] = k
            V[idx] = -1.0
        end
        for r in 1:rank
            idx += 1
            I[idx] = i
            J[idx] = idi+r
        end
        for r in 1:rank
            idx += 1
           I[idx] = i
           J[idx] = timei+r
       end
    end
    C = sparse(I, J, V)
    Ct = C'
    CtC = Ct * C

    # fill correctly sparse matrix
    Cvals = nonzeros(C)
    idx = zero(Int)
    for j in 1:n_regressors
        for i in 1:length(y)
            idx += 1
            Cvals[idx] = -X[i, j]
        end
    end

    f = (x, out) -> f2!(x, out, sqrtw, y, timerefs, idrefs, n_regressors, rank, Xt, N, T)
    g = (x, out)-> g2!(x, out, sqrtw, y, timerefs, idrefs, n_regressors, rank, Xt, N, T)

    df = DifferentiableSparseMultivariateFunction(f, g)
    result = nlsolve(df, x0)
    minimizer = result.minimum
    iterations = result.iterations
    converged =  result.x_converged || result.f_converged || result.gr_converged

    # expand minimumm -> (b, loadings and factors)
    b = minimizer[1:n_regressors]

    @inbounds for i in 1:length(idf.pool)
        idf.old1pool[i] =  x0[n_regressors + i]
    end
    @inbounds for i in 1:length(timef.pool)
        timef.old1pool[i] = x0[n_regressors + N * rank + i]
    end

    # rescale factors and loadings so that factors' * factors = Id
    rescale!(idf.pool, timef.pool, idf.old1pool, timef.old1pool)
    return (b, [iterations], [converged])
end


# fitness
function f2!{Tid, Ttime}(x::Vector{Float64}, out::Vector{Float64}, sqrtw::AbstractVector{Float64}, y::Vector{Float64}, timerefs::Vector{Ttime}, idrefs::Vector{Tid}, n_regressors::Integer, rank::Integer, Xt::Matrix{Float64}, N::Int, T::Int)
    @simd for i in 1:length(y)
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
        out[i] = y[i] - sqrtwi * prediction
    end
    return out
end




function g2!{Tid, Ttime}(x::Vector{Float64}, C::Base.SparseMatrix.SparseMatrixCSC{Float64, Int}, sqrtw::AbstractVector{Float64}, y::Vector{Float64}, timerefs::Vector{Ttime}, idrefs::Vector{Tid}, n_regressors::Integer, rank::Integer, Xt::Matrix{Float64}, N::Int, T::Int)

    Crows = rowvals(C)
    Cvals = nonzeros(C)
    icol = n_regressors 
    for j in 1:N
        for r in 1:rank
            icol += 1
            for k in nzrange(C, icol)
                row = Crows[k]
                Cvals[k] = - sqrtw[row] * x[timerefs[row]+r]
            end
        end
    end

    #sparse here
    for j in 1:T
         for r in 1:rank
            icol += 1
            for k in nzrange(C, icol)
                 row = Crows[k]
                 Cvals[k] = - sqrtw[row] * x[idrefs[row]+r]
             end
         end
     end
end





