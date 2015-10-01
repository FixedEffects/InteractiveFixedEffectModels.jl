##############################################################################
##
## Solution method: minimize by sparse least square optimization
##
##############################################################################

##############################################################################
##
## Factor Model (no regressors)
##
##############################################################################

function fit!(t::Union{Type{Val{:levenberg_marquardt}}, Type{Val{:dogleg}}}, 
                         fp::FactorModel,
                         fs::FactorSolution; 
                         maxiter::Integer = 10_000,
                         tol::Real = 1e-9,
                         lambda::Number = 0.0)
    # initialize
    iter = 0
    converged = true
    fp = FactorModel(deepcopy(fp.y), fp.sqrtw, fp.idrefs, fp.timerefs, rank(fp))
    N = size(fs.idpool, 1)
    T = size(fs.timepool, 1)

    fg = FactorGradient(fp,
                        FactorSolution(Array(Float64, N), Array(Float64, T)),
                        FactorSolution(Array(Float64, N), Array(Float64, T))
                       )
    nls = NonLinearLeastSquares(
                slice(fs, :, 1), 
                similar(fp.y), 
                fp, 
                fg, 
                g!)
    full = NonLinearLeastSquaresProblem(nls, t)
    for r in 1:rank(fp)
        fsr = slice(fs, :, r)
        full.nls.x = fsr
        result = optimize!(full,
            xtol = 1e-32, grtol = 1e-32, ftol = tol,  iterations = maxiter)
        iter += result.mul_calls
        converged = result.converged && converged
        if r < rank(fp)
            rescale!(fsr)
            subtract_factor!(fp, fsr)
        end
    end
    # rescale factors and loadings so that factors' * factors = Id
    return rescale(fs), iter, converged
end


##############################################################################
##
## Interactive FixedEffectModel
##
##############################################################################

function fit!(t::Union{Type{Val{:levenberg_marquardt}}, Type{Val{:dogleg}}},
                         fp::InteractiveFixedEffectsModel,
                         fs::InteractiveFixedEffectsSolution; 
                         maxiter::Integer = 10_000,
                         tol::Real = 1e-9,
                         lambda::Number = 0.0)
    timepoolT = 
    scaleb = vec(sumabs2(fp.X, 1))
    fs = InteractiveFixedEffectsSolution(fs.b, fs.idpool', fs.timepool')
    fg = InteractiveFixedEffectsGradient(fp, 
                InteractiveFixedEffectsSolution(similar(fs.b), similar(fs.idpool), similar(fs.timepool)),
                InteractiveFixedEffectsSolution(scaleb, similar(fs.idpool), similar(fs.timepool)))
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


##############################################################################
##
## Methods used in optimize! in LeastSquares
##
##############################################################################

function dot{T}(fs1::AbstractArray{T}, fs2::AbstractArray{T})  
    out = zero(typeof(one(T) * one(T)))
    @inbounds @simd for i in eachindex(fs1)
        out += fs1[i] * fs2[i]
    end
    return out
end

for t in (FactorSolution, InteractiveFixedEffectsSolution, HalfInteractiveFixedEffectsSolution)
    vars = [:(fill!(x.$field, α)) for field in fieldnames(t)]
    expr = Expr(:block, vars...)
    @eval begin
        function fill!(x::$t, α::Number)
             $expr
            return x
        end
    end

    vars = [:(scale!(x.$field, α)) for field in fieldnames(t)]
    expr = Expr(:block, vars...)
    @eval begin
        function scale!(x::$t, α::Number)
            $expr
            return x
        end
    end

    vars = [:(copy!(x1.$field, x2.$field)) for field in fieldnames(t)]
    expr = Expr(:block, vars...)
    @eval begin
        function copy!(x1::$t, x2::$t)
            $expr
            return x1
        end
    end

    vars = [:(axpy!(α, x1.$field, x2.$field)) for field in fieldnames(t)]
    expr = Expr(:block, vars...)
    @eval begin
        function axpy!(α::Number, x1::$t, x2::$t)
            $expr
            return x2
        end
    end

    vars = [:(map!(f, x1.$field, map(x -> x.$field, x2)...)) for field in fieldnames(t)]
    expr = Expr(:block, vars...)
    @eval begin
        function map!(f, x1::$t,  x2::$t...)
            $expr
            return x1
        end
    end
    vars = [:(dot(x1.$field, x2.$field)) for field in fieldnames(t)]
    expr = Expr(:call, :+, vars...)
    @eval begin
        function dot(x1::$t, x2::$t)
            $expr
        end
    end

    vars = [:(sumabs2(x.$field)) for field in fieldnames(t)]
    expr = Expr(:call, :+, vars...)
    @eval begin
        function sumabs2(x::$t)
            $expr
        end
    end

    vars = [:(length(x.$field)) for field in fieldnames(t)]
    expr = Expr(:call, :+, vars...)
    @eval begin
        function length(x::$t)
            $expr
        end
    end

    vars = [:(similar(x.$field)) for field in fieldnames(t)]
    expr = Expr(:call, t, vars...)
    @eval begin
        function similar(x::$t)
            $expr
        end
    end

    vars = [:(maxabs(x.$field)) for field in fieldnames(t)]
    expr = Expr(:call, :max, vars...)
    @eval begin
        function maxabs(x::$t)
            $expr
        end
    end
end

norm(fs::AbstractFactorSolution) = sqrt(sumabs2(fs))
eltype(fg::AbstractFactorSolution) = Float64

##############################################################################
##
## Factor Gradient
##
##############################################################################

type FactorGradient{Rank, W, Rid, Rtime, Tid, Ttime, sTid, sTtime} 
    fp::FactorModel{Rank, W, Rid, Rtime}
    fs::FactorSolution{1, Tid, Ttime}
    scalefs::FactorSolution{1, sTid, sTtime}
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

function Ac_mul_B!(α::Number, fg::FactorGradient, y::AbstractVector{Float64}, β::Number, fs::FactorSolution{1})
    mα = convert(Float64, -α)
    β == 0 ? fill!(fs, 0) : scale!(fs, β)
    @inbounds @simd for i in 1:length(y)
        sqrtwi = mα * y[i] * fg.fp.sqrtw[i]
        idi = fg.fp.idrefs[i]
        timei = fg.fp.timerefs[i]
        fs.idpool[idi] += sqrtwi * fg.fs.timepool[timei]
        fs.timepool[timei] += sqrtwi * fg.fs.idpool[idi]
    end
    return fs
end

function A_mul_B!(α::Number, fg::FactorGradient, fs::FactorSolution{1}, β::Number, y::AbstractVector{Float64})
    mα = convert(Float64, -α)
    β == 0 ? fill!(y, 0) : scale!(y, β)
    @fastmath @inbounds @simd for i in 1:length(y)
        timei = fg.fp.timerefs[i]
        idi = fg.fp.idrefs[i]
        out = (fg.fs.idpool[idi] * fs.timepool[timei] 
            + fg.fs.timepool[timei] * fs.idpool[idi])
        y[i] += mα * fg.fp.sqrtw[i] * out
    end
end

function call(fp::FactorModel, fs::FactorSolution{1}, out::AbstractVector{Float64})
    copy!(out, fp.y)
    @fastmath @inbounds @simd for i in 1:length(out)
        sqrtwi = fp.sqrtw[i]
        idi = fp.idrefs[i]
        timei = fp.timerefs[i]
        out[i] -= sqrtwi * fs.idpool[idi] * fs.timepool[timei]
    end
    return out
end

function g!(fs::FactorSolution{1}, fg::FactorGradient)
    copy!(fg.fs, fs)
    fill!(fg.scalefs, zero(Float64))
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
## InteractiveFixedEffectsGradient
##
##############################################################################

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
        β == 0 ? fill!(fs, 0) : scale!(fs, β)
        @inbounds @simd for k in 1:length(fs.b)
            out = zero(Float64)
             for i in 1:length(y)
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
        β == 0 ? fill!(y, 0) : scale!(y, β)
        Base.BLAS.gemm!('N', 'N', mα, fg.fp.X, fs.b, 1.0, y)
        @fastmath @inbounds @simd for i in 1:length(y)
            timei = fg.fp.timerefs[i]
            idi = fg.fp.idrefs[i]
            out = 0.0
            @nexprs $Rank r -> begin
                 out += (fg.fs.idpool[r, idi] * fs.timepool[r, timei] 
                         + fg.fs.timepool[r, timei] * fs.idpool[r, idi])
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
        copy!(fg.fs, fs)

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

