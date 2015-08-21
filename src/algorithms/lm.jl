
##############################################################################
##
## Estimate interactive factor model by incremental optimization routine
##
##############################################################################

function fit!{Rid, Rtime}(::Type{Val{:lm}}, 
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


    f = x -> f!(x, similar(y), sqrtw, y, timerefs, idrefs, n_regressors, rank, Xt, N, T)
    g = (x, a, at, ata)-> g!(x, a, at, ata, sqrtw, y, timerefs, idrefs, n_regressors, rank, Xt, N, T)

    # convergence is chebyshev for x
    result = levenberg_marquardt2(f, g, x0, C, Ct, CtC)
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
function f!{Tid, Ttime}(x::Vector{Float64}, out::Vector{Float64}, sqrtw::AbstractVector{Float64}, y::Vector{Float64}, timerefs::Vector{Ttime}, idrefs::Vector{Tid}, n_regressors::Integer, rank::Integer, Xt::Matrix{Float64}, N::Int, T::Int)
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




function g!{Tid, Ttime}(x::Vector{Float64}, C::Base.SparseMatrix.SparseMatrixCSC{Float64, Int}, Ct::Base.SparseMatrix.SparseMatrixCSC{Float64, Int}, CtC::Base.SparseMatrix.SparseMatrixCSC{Float64, Int}, sqrtw::AbstractVector{Float64}, y::Vector{Float64}, timerefs::Vector{Ttime}, idrefs::Vector{Tid}, n_regressors::Integer, rank::Integer, Xt::Matrix{Float64}, N::Int, T::Int)

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


    Ctrows = rowvals(Ct)
    Ctvals = nonzeros(Ct)
    idx = zero(Int)
    for i in 1:length(y)
        sqrtwi = sqrtw[i]
        idi = idrefs[i]
        timei = timerefs[i]
        for k in 1:n_regressors
            idx += 1
            Ctvals[idx] = -Xt[k, i]
        end
        for r in 1:rank
            idx += 1
            Ctvals[idx] = - sqrtwi * x[timei+r]
        end
        for r in 1:rank
            idx += 1
            Ctvals[idx] = - sqrtwi * x[idi+r]
        end
    end
    A_mul_B2!(CtC, Ct, C)
end




function levenberg_marquardt2(f::Function, g::Function, x0, C, Ct, CtC; tolX=1e-8, tolG=1e-12, maxIter=100, lambda=100.0, show_trace=false, )
    const MAX_LAMBDA = 1e16 # minimum trust region radius
    const MIN_LAMBDA = 1e-16 # maximum trust region radius
    const MIN_STEP_QUALITY = 1e-3
    const MIN_DIAGONAL = 1e-6 # lower bound on values of diagonal matrix used to regularize the trust region step

    converged = false
    x_converged = false
    g_converged = false
    need_jacobian = true
    iterCt = 0
    x = x0
    delta_x = copy(x0)
    f_calls = 0
    g_calls = 0

    fcur = f(x)
    fcur2 = similar(fcur)
    f_calls += 1
    residual = sumabs2(fcur)


    summ = fill(zero(Float64), length(x0))

    # Maintain a trace of the system.
    tr = Optim.OptimizationTrace()
    if show_trace
        d = @compat Dict("lambda" => lambda)
        os = Optim.OptimizationState(iterCt, sumabs2(fcur), NaN, d)
        push!(tr, os)
        println(os)
    end
    while ( ~converged && iterCt < maxIter )
        if need_jacobian
                g(x, C, Ct, CtC)
            g_calls += 1
            need_jacobian = false
        end
        fill!(summ, zero(Float64))
        Cvals = nonzeros(C)
        for j in 1:size(C,2)
            for k in nzrange(C, j)
                summ[j] += Cvals[k]^2
            end
        end
        DtD = Float64[max(x, MIN_DIAGONAL) for x in summ]
       
        sqrtl = sqrt(lambda)
        for i in 1:size(CtC, 1)
            CtC[i, i] += sqrtl * DtD[i]
        end
        delta_x = CtC \ (-Ct * fcur)
        predicted_residual = sumabs2(C*delta_x + fcur)
        if predicted_residual > residual + 2max(eps(predicted_residual),eps(residual))
            warn("""Problem solving for delta_x: predicted residual increase.
                             $predicted_residual (predicted_residual) >
                             $residual (residual) + $(eps(predicted_residual)) (eps)""")
        end
        trial_f = f(x + delta_x)
        f_calls += 1
        trial_residual = sumabs2(trial_f)
        rho = (trial_residual - residual) / (predicted_residual - residual)

        if rho > MIN_STEP_QUALITY
            x += delta_x
            fcur = trial_f
            residual = trial_residual
            # increase trust region radius
            lambda = max(0.1*lambda, MIN_LAMBDA)
            need_jacobian = true
        else
            # decrease trust region radius
            lambda = min(10*lambda, MAX_LAMBDA)
        end
        iterCt += 1

        # show state
        if show_trace
            gradnorm = norm(Ct*fcur, Inf)
            d = @compat Dict("g(x)" => gradnorm, "dx" => delta_x, "lambda" => lambda)
            os = Optim.OptimizationState(iterCt, sumabs2(fcur), gradnorm, d)
            push!(tr, os)
            println(os)
        end
        if norm(Ct * fcur, Inf) < tolG
            g_converged = true
        elseif norm(delta_x) < tolX*(tolX + norm(x))
            x_converged = true
        end
        converged = g_converged | x_converged   
    end

    Optim.MultivariateOptimizationResults("Levenberg-Marquardt", x0, x, sumabs2(fcur), iterCt, !converged, x_converged, 0.0, false, 0.0, g_converged, tolG, tr, f_calls, g_calls)
end



function A_mul_B2!{Tv,Ti}(C::SparseMatrixCSC{Tv,Ti}, A::SparseMatrixCSC{Tv,Ti}, B::SparseMatrixCSC{Tv,Ti}; sortindices::Symbol = :sortcols)
    mA, nA = size(A)
    mB, nB = size(B)
    nA==mB || throw(DimensionMismatch())

    colptrA = A.colptr; rowvalA = A.rowval; nzvalA = A.nzval
    colptrB = B.colptr; rowvalB = B.rowval; nzvalB = B.nzval
    # TODO: Need better estimation of result space
    colptrC = C.colptr
    rowvalC = C.rowval
    nzvalC = C.nzval
    nnzC = length(colptrC)

    @inbounds begin
        ip = 1
        xb = zeros(Ti, mA)
        x  = zeros(Tv, mA)
        for i in 1:nB
            if ip + mA - 1 > nnzC
                resize!(rowvalC, nnzC + max(nnzC,mA))
                resize!(nzvalC, nnzC + max(nnzC,mA))
                nnzC = length(nzvalC)
            end
            colptrC[i] = ip
            for jp in colptrB[i]:(colptrB[i+1] - 1)
                nzB = nzvalB[jp]
                j = rowvalB[jp]
                for kp in colptrA[j]:(colptrA[j+1] - 1)
                    nzC = nzvalA[kp] * nzB
                    k = rowvalA[kp]
                    if xb[k] != i
                        rowvalC[ip] = k
                        ip += 1
                        xb[k] = i
                        x[k] = nzC
                    else
                        x[k] += nzC
                    end
                end
            end
            for vp in colptrC[i]:(ip - 1)
                nzvalC[vp] = x[rowvalC[vp]]
            end
        end
        colptrC[nB+1] = ip
    end
    # The Gustavson algorithm does not guarantee the product to have sorted row indices.
    Base.SparseMatrix.sortSparseMatrixCSC!(C, sortindices=sortindices)
end



