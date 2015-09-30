##############################################################################
##
## Solution method: minimize by sparse least square optimization
##
##############################################################################

##############################################################################
##
## Common methods
##
##############################################################################
eltype(fg::AbstractFactorSolution) = Float64
 function similar(fs::FactorSolution)
    return FactorSolution(similar(fs.idpool), similar(fs.timepool))
end
function similar(fs::InteractiveFixedEffectsSolution)
     return InteractiveFixedEffectsSolution(similar(fs.b), similar(fs.idpool), similar(fs.timepool))
 end

function length(fs::InteractiveFixedEffectsSolution)
    return length(fs.b) + length(fs.idpool) + length(fs.timepool)
end
function length(fs::FactorSolution)
    return length(fs.idpool) + length(fs.timepool)
end

function sumabs2(fs::InteractiveFixedEffectsSolution)
    sumabs2(fs.b) + sumabs2(fs.idpool) + sumabs2(fs.timepool)
end
function sumabs2(fs::FactorSolution)
    sumabs2(fs.idpool) + sumabs2(fs.timepool)
end

norm(fs::AbstractFactorSolution) = sqrt(sumabs2(fs))

function maxabs(fs::InteractiveFixedEffectsSolution)
    max(maxabs(fs.b), maxabs(fs.idpool), maxabs(fs.timepool))
end
function maxabs(fs::FactorSolution)
    max(maxabs(fs.idpool), maxabs(fs.timepool))
end


@generated function fill!(x::AbstractFactorSolution, α::Number)
    vars = [:(fill!(x.$field, α)) for field in fieldnames(x)]
    Expr(:block, vars..., :(return x))
end

@generated function scale!(x::AbstractFactorSolution, α::Number)
    vars = [:(scale!(x.$field, α)) for field in fieldnames(x)]
    Expr(:block, vars..., :(return x))
end


@generated function copy!(x1::AbstractFactorSolution, x2::AbstractFactorSolution)
    vars = [:(copy!(x1.$field, x2.$field)) for field in fieldnames(x1)]
    Expr(:block, vars..., :(return x1))
end


@generated function axpy!(α::Number, x1::AbstractFactorSolution, x2::AbstractFactorSolution)
    vars = [:(axpy!(α, x1.$field, x2.$field)) for field in fieldnames(x1)]
    Expr(:block, vars..., :(return x2))
end
       

@generated function map!(f, x1::AbstractFactorSolution,  x2::AbstractFactorSolution...)
   vars = [:(map!(f, x1.$field, map(x -> x.$field, x2)...)) for field in fieldnames(x1)]
    Expr(:block, vars..., :(return x1))
end

function dot{T}(fs1::AbstractArray{T}, fs2::AbstractArray{T})  
    out = zero(T)
    @inbounds @simd for i in eachindex(fs1)
        out += fs1[i] * fs2[i]
    end
    return out
end

@generated function dot(x1::AbstractFactorSolution, x2::AbstractFactorSolution)
    expr1 = :(out = zero(eltype(x1)))
    vars = [:(out += dot(x1.$field, x2.$field)) for field in fieldnames(x1)]
    Expr(:block, expr1, vars..., :(return out))
end




##############################################################################
##
## Factor Model (no regressors)
##
##############################################################################

function fit!(t::Union{Type{Val{:levenberg_marquardt}}, Type{Val{:dogleg}}}, 
                         fp::FactorModel,
                         fs::FactorSolution; 
                         maxiter::Integer = 100_000,
                         tol::Real = 1e-9,
                         lambda::Real = 0.0)
    # initialize
    iter = 0
    converged = true
    fp = FactorModel(deepcopy(fp.y), fp.sqrtw, fp.idrefs, fp.timerefs, rank(fp))
    idpoolr = slice(fs.idpool, :, 1)
    timepoolr = slice(fs.timepool, :, 1)
    fg = FactorGradient(fp,
                        FactorSolution(similar(idpoolr), similar(timepoolr)),
                        FactorSolution(similar(idpoolr), similar(timepoolr))
                       )
    nls = NonLinearLeastSquares(
                FactorSolution(idpoolr, timepoolr), 
                similar(fp.y), 
                fp, 
                fg, 
                g!)
    full = NonLinearLeastSquaresProblem(nls, t)
    for r in 1:rank(fp)
        fsr = FactorSolution(slice(fs.idpool, :, r), slice(fs.timepool, :, r))
        full.nls.x = fsr
        result = optimize!(full,
            xtol = 1e-32, grtol = 1e-32, ftol = tol,  iterations = maxiter)
        iter += result.mul_calls
        converged = result.converged && converged
        if r < rank(fp)
            rescale!(fsr)
            subtract_factor!(fp.y, fp, fsr)
        end
    end
    # rescale factors and loadings so that factors' * factors = Id
    return rescale(fs), iter, converged
end

type FactorGradient{Rank, W, Rid, Rtime, Tid, Ttime, sTid, sTtime} 
    fp::FactorModel{Rank, W, Rid, Rtime}
    fs::FactorSolution{Tid, Ttime}
    scalefs::FactorSolution{sTid, sTtime}
end
Base.rank{Rank}(f::FactorGradient{Rank}) = Rank


function size(fg::FactorGradient, i::Integer)
    if i == 1
        length(fg.fp.y)
    elseif i == 2
      length(fg.fs.idpool) + length(fg.fs.timepool)
    end
end
size(fg::FactorGradient) = (size(fg, 1), size(fg, 2))
eltype(fg::FactorGradient) = Float64


function colsumabs2!(fs::FactorSolution, fg::FactorGradient) 
    copy!(fs, fg.scalefs)
end

function Ac_mul_B!(α::Number, fg::FactorGradient, y::AbstractVector{Float64}, β::Number, fs::FactorSolution)
    mα = convert(Float64, -α)
    if β != 1
        if β == 0
            fill!(fs, 0)
        else
            scale!(fs, β)
        end
    end
    @inbounds @simd for i in 1:length(y)
        sqrtwi = mα * y[i] * fg.fp.sqrtw[i]
        idi = fg.fp.idrefs[i]
        timei = fg.fp.timerefs[i]
        fs.idpool[idi] += sqrtwi * fg.fs.timepool[timei]
        fs.timepool[timei] += sqrtwi * fg.fs.idpool[idi]
    end
    return fs
end

function A_mul_B!(α::Number, fg::FactorGradient, fs::FactorSolution, β::Number, y::AbstractVector{Float64})
    mα = convert(Float64, -α)
    if β != 1
        if β == 0
            fill!(y, 0)
        else
            scale!(y, β)
        end
    end
    @fastmath @inbounds @simd for i in 1:length(y)
        timei = fg.fp.timerefs[i]
        idi = fg.fp.idrefs[i]
        out = (fg.fs.idpool[idi] * fs.timepool[timei] 
                               + fg.fs.timepool[timei] * fs.idpool[idi]
                               )
        y[i] += mα * fg.fp.sqrtw[i] * out
    end
    return y
end

function call(fp::FactorModel, fs::FactorSolution, out::Vector{Float64})
    copy!(out, fp.y)
    @fastmath @inbounds @simd for i in 1:length(out)
        sqrtwi = fp.sqrtw[i]
        idi = fp.idrefs[i]
        timei = fp.timerefs[i]
        out[i] -= sqrtwi * fs.idpool[idi] * fs.timepool[timei]
    end
    return out
end

function g!(fs::FactorSolution, fg::FactorGradient)
    copy!(fg.fs.idpool, fs.idpool)
    copy!(fg.fs.timepool, fs.timepool)
    fill!(fg.scalefs.idpool, zero(Float64))
    fill!(fg.scalefs.timepool, zero(Float64))
    @inbounds @simd for i in 1:length(fg.fp.y)
        sqrtwi = fg.fp.sqrtw[i]
        idi = fg.fp.idrefs[i]
        timei = fg.fp.timerefs[i] 
        fg.scalefs.idpool[idi] += abs2(sqrtwi * fg.fs.timepool[timei])
        fg.scalefs.timepool[timei] += abs2(sqrtwi * fg.fs.idpool[idi])
    end
    return fg
end
##############################################################################
##
## Interactive FixedEffectModel
##
##############################################################################

function fit!(t::Union{Type{Val{:levenberg_marquardt}}, Type{Val{:dogleg}}},
                         fp::InteractiveFixedEffectsModel,
                         fs::InteractiveFixedEffectsSolution; 
                         maxiter::Integer = 100_000,
                         tol::Real = 1e-9,
                         lambda::Real = 0.0)
    idpoolT = fs.idpool'
    timepoolT = fs.timepool'
    scaleb = vec(sumabs2(fp.X, 1))
    fs = InteractiveFixedEffectsSolution(fs.b, idpoolT, timepoolT)
    fg = InteractiveFixedEffectsGradient(fp, 
                InteractiveFixedEffectsSolution(similar(fs.b), similar(idpoolT), similar(timepoolT)),
                InteractiveFixedEffectsSolution(scaleb, similar(idpoolT), similar(timepoolT)))
    nls = NonLinearLeastSquares(fs, similar(fp.y), fp, fg, g!)
    temp = similar(fp.y)
    if t == Val{:levenberg_marquardt}
        result = optimize!(nls ; method = :levenberg_marquardt,
            xtol = 1e-32, grtol = 1e-32, ftol = tol,  iterations = maxiter)
    else
          result = optimize!(nls ; method = :dogleg,
          xtol = 1e-32, grtol = 1e-32, ftol = tol,  iterations = maxiter)
    end
    # rescale factors and loadings so that factors' * factors = Id
    fs = InteractiveFixedEffectsSolution(fs.b, fs.idpool', fs.timepool')
    return rescale(fs), result.mul_calls, result.converged
end

type InteractiveFixedEffectsGradient{Rank, W, Rid, Rtime, Tb, Tid, Ttime, sTb, sTid, sTtime} 
    fp::InteractiveFixedEffectsModel{Rank, W, Rid, Rtime}
    fs::InteractiveFixedEffectsSolution{Tb, Tid, Ttime}
    scalefs::InteractiveFixedEffectsSolution{sTb, sTid, sTtime}
end

rank{Rank}(f::InteractiveFixedEffectsGradient{Rank}) = Rank

function size(fg::InteractiveFixedEffectsGradient, i::Integer)
    if i == 1
        length(fg.fp.y)
    elseif i == 2
      length(fg.fs.b) + length(fg.fs.idpool) + length(fg.fs.timepool)
    end
end
size(fg::InteractiveFixedEffectsGradient) = (size(fg, 1), size(fg, 2))

eltype(fg::InteractiveFixedEffectsGradient) = Float64

function colsumabs2!(fs::InteractiveFixedEffectsSolution, fg::InteractiveFixedEffectsGradient) 
    copy!(fs, fg.scalefs)
end

@generated function Ac_mul_B!{Rank}(α::Number, fg::InteractiveFixedEffectsGradient{Rank}, y::AbstractVector{Float64}, β::Number, fs::InteractiveFixedEffectsSolution)
    quote
        mα = convert(Float64, -α)
        if β != 1
            if β == 0
                fill!(fs, 0)
            else
                scale!(fs, β)
            end
        end
        for k in 1:length(fs.b)
            out = zero(Float64)
            @inbounds @simd for i in 1:length(y)
                out += y[i] * fg.fp.X[i, k]
            end
            fs.b[k] += mα * out
        end
        @inbounds @simd for i in 1:length(y)
            sqrtwi = mα * y[i] * fg.fp.sqrtw[i]
            idi = fg.fp.idrefs[i]
            timei = fg.fp.timerefs[i]
            @nexprs $Rank r -> begin
                fs.idpool[r, idi] += sqrtwi * fg.fs.timepool[r, timei]
                fs.timepool[r, timei] += sqrtwi * fg.fs.idpool[r, idi]
            end
        end
        return fs
    end
end

@generated function A_mul_B!{Rank}(α::Number, fg::InteractiveFixedEffectsGradient{Rank}, fs::InteractiveFixedEffectsSolution, β::Number, y::AbstractVector{Float64})
    quote
        mα = convert(Float64, -α)
        if β != 1
            if β == 0
                fill!(y, 0)
            else
                scale!(y, β)
            end
        end
        Base.BLAS.gemm!('N', 'N', mα, fg.fp.X, fs.b, 1.0, y)
        @fastmath @inbounds @simd for i in 1:length(y)
            timei = fg.fp.timerefs[i]
            idi = fg.fp.idrefs[i]
            out = 0.0
            @nexprs $Rank r -> begin
                 out += (fg.fs.idpool[r, idi] * fs.timepool[r, timei] 
                         + fg.fs.timepool[r, timei] * fs.idpool[r, idi]
                         )
             end
            y[i] += mα * fg.fp.sqrtw[i] * out
        end
        return y
    end
end

@generated function call{Rank}(fp::InteractiveFixedEffectsModel{Rank}, 
    fs::InteractiveFixedEffectsSolution, out::Vector{Float64})
    quote
        copy!(out, fp.y)
        BLAS.gemm!('N', 'N', -1.0, fp.X, fs.b, 1.0, out)
        @fastmath @inbounds @simd for i in 1:length(out)
            sqrtwi = fp.sqrtw[i]
            idi = fp.idrefs[i]
            timei = fp.timerefs[i]
            @nexprs $Rank r -> begin
                out[i] -= sqrtwi * fs.idpool[r, idi] * fs.timepool[r, timei]
            end
        end
        return out
    end
end

@generated function g!{Rank}(fs::InteractiveFixedEffectsSolution, fg::InteractiveFixedEffectsGradient{Rank})
    quote
        copy!(fg.fs.b, fs.b)
        copy!(fg.fs.idpool, fs.idpool)
        copy!(fg.fs.timepool, fs.timepool)

        # fill scale
        fill!(fg.scalefs.idpool, zero(Float64))
        fill!(fg.scalefs.timepool, zero(Float64))
        @inbounds @simd for i in 1:length(fg.fp.y)
            sqrtwi = fg.fp.sqrtw[i]
            idi = fg.fp.idrefs[i]
            timei = fg.fp.timerefs[i] 
            @nexprs $Rank r -> begin
                fg.scalefs.idpool[r, idi] += abs2(sqrtwi * fg.fs.timepool[r, timei])
                fg.scalefs.timepool[r, timei] += abs2(sqrtwi * fg.fs.idpool[r, idi])
            end
        end
    end
end

