##############################################################################
##
## Solution method: coordinate gradient descent 
## (also called alternative regressions, Gauss Seidel)
##
##############################################################################

##############################################################################
##
## Factor Model (no regressors)
##
##############################################################################

function fit!(::Type{Val{:gauss_seidel}}, 
    fp::FactorModel,
    fs::FactorSolution;
    maxiter::Integer  = 100_000,
    tol::Real = 1e-9,
    lambda::Real = 0.0
    )
    lambda == 0.0 || error("The alternative regression method only works with lambda = 0.0")

    # initialize
    iter = 0
    converged = true

    fp = FactorModel(copy(fp.y), fp.sqrtw, fp.idrefs, fp.timerefs, rank(fp))
    idscale = Array{Float64}(undef, size(fs.idpool, 1))
    timescale = Array{Float64}(undef, size(fs.timepool, 1))

    for r in 1:rank(fp)
        fsr = view(fs, :, r)
        oldf_x = ssr(fp, fsr)
        iter_inner = 0
        while iter_inner < maxiter
            iter_inner += 1
            update!(Val{:gauss_seidel}, fp.y, fp.sqrtw, fp.idrefs, fp.timerefs, fsr.idpool, 
                idscale, fsr.timepool)
            update!(Val{:gauss_seidel}, fp.y, fp.sqrtw, fp.timerefs, fp.idrefs, fsr.timepool, 
                timescale, fsr.idpool)
            f_x = ssr(fp, fsr)
            if abs(f_x - oldf_x) < (abs(f_x) + tol) * tol
                break
            else
                oldf_x = f_x
            end
        end
        iter = max(iter_inner, iter)
        converged = (iter_inner < maxiter) && converged
        if r < rank(fp)
            rescale!(fsr)
            subtract_factor!(fp, fsr)
        end
    end
    return rescale(fs), iter, converged
end

##############################################################################
##
## Interactive FixedEffectModel
##
##############################################################################

function fit!(::Type{Val{:gauss_seidel}},
    fp::InteractiveFixedEffectsModel,
    fs::InteractiveFixedEffectsSolution; 
    maxiter::Integer = 100_000, 
    tol::Real = 1e-9,
    lambda::Real = 0.0)
    lambda == 0.0 || error("The alternative regression method only works with lambda = 0.0")


    N = size(fs.idpool, 1)
    T = size(fs.timepool, 1)

    #qr fact factorization cannot divide in place for now
    crossx = cholesky!(fp.X' * fp.X)
    M = crossx \ fp.X'

    yoriginal = copy(fp.y)
    fp = InteractiveFixedEffectsModel(copy(fp.y), fp.sqrtw, fp.X, fp.idrefs, fp.timerefs, rank(fp))

    idscale = Array{Float64}(undef, size(fs.idpool, 1))
    timescale = Array{Float64}(undef, size(fs.timepool, 1))

    # starts loop
    converged = false
    iterations = maxiter
    iter = 0
    f_x = Inf
    oldf_x = Inf
    while iter < maxiter
        iter += 1
        (f_x, oldf_x) = (oldf_x, f_x)

        # Given beta, compute incrementally an approximate factor model
        copyto!(fp.y, yoriginal)
        gemm!('N', 'N', -1.0, fp.X, fs.b, 1.0, fp.y)
        for r in 1:rank(fp)
            fsr = view(fs, :, r)
            update!(Val{:gauss_seidel}, fp.y, fp.sqrtw, fp.idrefs, fp.timerefs, fsr.idpool, idscale, fsr.timepool)
            update!(Val{:gauss_seidel}, fp.y, fp.sqrtw, fp.timerefs, fp.idrefs, fsr.timepool, timescale, fsr.idpool)
            subtract_factor!(fp, fsr)
        end

        # Given factor model, compute beta
        copyto!(fp.y, yoriginal)
        subtract_factor!(fp, fs)
        ## corresponds to Gauss Niedel with acceleration
        rmul!(fs.b, -0.5)
        gemm!('N', 'N', 1.5, M, fp.y, 1.0, fs.b)

        # Check convergence
        gemm!('N', 'N', -1.0, fp.X, fs.b, 1.0, fp.y)
        f_x = sum(abs2, fp.y)
        if abs(f_x - oldf_x) < (abs(f_x) + tol) * tol
            converged = true
            iterations = iter
            break
        end
    end
    return rescale(fs), iterations, converged
end

##############################################################################
##
## update! by alternative regressions
##
##############################################################################

function update!(::Type{Val{:gauss_seidel}},
    y::Vector{Float64},
    sqrtw::AbstractVector{Float64},
    p1refs::Vector{R1},
    p2refs::Vector{R2},
    p1::AbstractVector{Float64},
    p1scale::Vector{Float64},
    p2::AbstractVector{Float64}
    ) where {R1, R2}
    fill!(p1, zero(Float64))
    fill!(p1scale, zero(Float64))
    @inbounds @simd for i in 1:length(y)
        p1i = p1refs[i]
        xi = sqrtw[i] * p2[p2refs[i]] 
        p1[p1i] += xi * y[i]
        p1scale[p1i] += abs2(xi)
    end
    @inbounds @simd for i in 1:length(p1)
        s = p1scale[i]
        if s > 0
            p1[i] /= s
        end
    end
end

## compute sum of squared residuals
function ssr(fm::FactorModel, fs::FactorSolution{1})
    out = zero(Float64)
    @inbounds @simd for i in 1:length(fm.y)
        out += abs2(fm.y[i] - fm.sqrtw[i] * fs.idpool[fm.idrefs[i]] * fs.timepool[fm.timerefs[i]])
    end 
    return out
end
