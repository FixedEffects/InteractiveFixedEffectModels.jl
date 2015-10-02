##############################################################################
##
## Solution method: minimize by sparse least square optimization
##
##############################################################################


##############################################################################
##
## Interactive FixedEffectsModel
##
##############################################################################

function fit!(t::Type{Val{:regar}},
                         fp::InteractiveFixedEffectsModel,
                         fs::InteractiveFixedEffectsSolution; 
                         maxiter::Integer = 10_000,
                         tol::Real = 1e-9,
                         lambda::Number = 0.0)

    scaleb = vec(sumabs2(fp.X, 1))
    fs = InteractiveFixedEffectsSolution(fs.b,  fs.idpool',fs.timepool')

    res = deepcopy(fp.y)

    fsid = HalfInteractiveFixedEffectsSolution(fs.b, fs.idpool)
    solveid = NTuple((similar(fsid), similar(fsid), similar(fsid), similar(fsid), similar(fsid)))
    fpid = HalfInteractiveFixedEffectsModel(fp.y, fp.sqrtw, fp.X, fp.idrefs, fp.timerefs, fs.timepool, (length(fp.y), size(fs.idpool, 2) * size(fs.idpool, 1) + length(fs.b)), rank(fp))

    fstime = HalfInteractiveFixedEffectsSolution(fs.b, fs.timepool)
    solvetime = NTuple((similar(fstime), similar(fstime), similar(fstime), similar(fstime), similar(fstime)))
    fptime = HalfInteractiveFixedEffectsModel(fp.y, fp.sqrtw, fp.X, fp.timerefs, fp.idrefs, fs.idpool, (length(fp.y), size(fs.timepool, 2) * size(fs.timepool, 1) + length(fs.b)), rank(fp))
    iter = 0


    oldb = deepcopy(fs.b)
    while iter <= maxiter
        iter += 1
        # regress on X and factors
        lsid = LinearLeastSquaresProblem(
            LinearLeastSquares(fsid, copy!(res, fp.y), fpid),
            solveid)
        x, ch = optimize!(lsid)
        iter += ch.mvps
        # regress on X and loadings
        lstime = LinearLeastSquaresProblem(
            LinearLeastSquares(fstime, copy!(res, fp.y), fptime),
            solvetime)
        x, ch = optimize!(lstime)
        iter += ch.mvps
        if chebyshev(fs.b, oldb) <= tol
            break
        end
        copy!(oldb, fs.b)
    end
    # rescale factors and loadings so that factors' * factors = Id
    fs = InteractiveFixedEffectsSolution(fs.b, fs.idpool', fs.timepool')
    return rescale(fs), iter, iter < maxiter
end


##############################################################################
##
## Methods used in optimize! in LeastSquares
##
##############################################################################

rank{Rank}(f::HalfInteractiveFixedEffectsModel{Rank}) = Rank

size(fg::HalfInteractiveFixedEffectsModel) = fg.size
size(fg::HalfInteractiveFixedEffectsModel, i::Integer) = fg.size[i]
eltype(fg::HalfInteractiveFixedEffectsModel) = Float64

@generated function colsumabs2!{Rank}(fs::HalfInteractiveFixedEffectsSolution, fp::HalfInteractiveFixedEffectsModel{Rank})
    quote 
        for i in 1:length(fs.b)
            fs.b[i] = sumabs2(slice(fp.X, :, i))
        end

        fill!(fs.idpool, zero(Float64))
        @inbounds @simd for i in 1:length(fp.y)
            sqrtwi = fp.sqrtw[i]
            idi = fp.idrefs[i]
            timei = fp.timerefs[i] 
            @nexprs $Rank r -> begin
                fs.idpool[r, idi] += abs2(sqrtwi * fp.timepool[r, timei])
            end
        end
    end
end

@generated function Ac_mul_B!{Rank}(α::Number, fp::HalfInteractiveFixedEffectsModel{Rank}, y::AbstractVector{Float64}, β::Number, fs::HalfInteractiveFixedEffectsSolution)
    quote
        mα = convert(Float64, α)
        safe_scale!(fs, β)
        @inbounds @simd for k in 1:length(fs.b)
            out = zero(Float64)
             for i in 1:length(y)
                out += y[i] * fp.X[i, k]
            end
            fs.b[k] += mα * out
        end
        @inbounds @simd for i in 1:length(y)
            sqrtwi = mα * y[i] * fp.sqrtw[i]
            idi = fp.idrefs[i]
            timei = fp.timerefs[i]
            @nexprs $Rank r -> begin
                fs.idpool[r, idi] += sqrtwi * fp.timepool[r, timei]
            end
        end
        return fs
    end
end

@generated function A_mul_B!{Rank}(α::Number, fp::HalfInteractiveFixedEffectsModel{Rank}, fs::HalfInteractiveFixedEffectsSolution, β::Number, y::AbstractVector{Float64})
    quote
        mα = convert(Float64, α)
        safe_scale!(y, β)
        Base.BLAS.gemm!('N', 'N', mα, fp.X, fs.b, 1.0, y)
        @fastmath @inbounds @simd for i in 1:length(y)
            timei = fp.timerefs[i]
            idi = fp.idrefs[i]
            out = 0.0
            @nexprs $Rank r -> begin
                 out += fp.timepool[r, timei] * fs.idpool[r, idi]
             end
            y[i] += mα * fp.sqrtw[i] * out
        end
        return y
    end
end


