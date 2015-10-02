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
    fullrank = rank(fp)
    fp = FactorModel(deepcopy(fp.y), fp.sqrtw, fp.idrefs, fp.timerefs, 1)
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
    for r in 1:fullrank
        fsr = slice(fs, :, r)
        full.nls.x = fsr
        result = optimize!(full,
            xtol = 1e-32, grtol = 1e-32, ftol = tol,  iterations = maxiter)
        iter += result.mul_calls
        converged = result.converged && converged
        if r < fullrank 
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

function fit!{Rank}(t::Union{Type{Val{:levenberg_marquardt}}, Type{Val{:dogleg}}},
                         fp::InteractiveFixedEffectsModel{Rank},
                         fs::InteractiveFixedEffectsSolution{Rank}; 
                         maxiter::Integer = 10_000,
                         tol::Real = 1e-9,
                         lambda::Number = 0.0)
    timepoolT = 
    scaleb = vec(sumabs2(fp.X, 1))
    fsT = InteractiveFixedEffectsSolutionT(fs.b, fs.idpool', fs.timepool')
    fg = InteractiveFixedEffectsGradientT(fp, 
                similar(fsT),
                InteractiveFixedEffectsSolutionT(scaleb, similar(fsT.idpool), similar(fsT.timepool)))
    nls = NonLinearLeastSquares(fsT, similar(fp.y), fp, fg, g!)
    temp = similar(fp.y)
    if t == Val{:levenberg_marquardt}
        result = optimize!(nls ; method = :levenberg_marquardt,
            xtol = 1e-32, grtol = 1e-32, ftol = tol,  iterations = maxiter)
    else
          result = optimize!(nls ; method = :dogleg,
          xtol = 1e-32, grtol = 1e-32, ftol = tol,  iterations = maxiter)
    end
    # rescale factors and loadings so that factors' * factors = Id
    fs = InteractiveFixedEffectsSolution(fsT.b, fsT.idpool', fsT.timepool')
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

for t in (FactorSolution, InteractiveFixedEffectsSolution, InteractiveFixedEffectsSolutionT, HalfInteractiveFixedEffectsSolution)
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

function safe_scale!(x, β)
    if β != 1
        β == 0 ? fill!(x, zero(eltype(x))) : scale!(x, β)
    end
end
##############################################################################
##
## Factor Gradient
##
##############################################################################
abstract AbstractFactorGradient{T}

type FactorGradient{Rank, W, Rid, Rtime, Tid, Ttime, sTid, sTtime} <: AbstractFactorGradient{Rank}
    fp::FactorModel{Rank, W, Rid, Rtime}
    fs::FactorSolution{1, Tid, Ttime}
    scalefs::FactorSolution{1, sTid, sTtime}
end

function size(fg::FactorGradient, i::Integer)
    if i == 1
        length(fg.fp.y)
    elseif i == 2
      length(fg.fs.idpool) + length(fg.fs.timepool)
    end
end

type InteractiveFixedEffectsGradientT{Rank, W, Rid, Rtime, Tb, Tid, Ttime, sTb, sTid, sTtime} <: AbstractFactorGradient{Rank}
    fp::InteractiveFixedEffectsModel{Rank, W, Rid, Rtime}
    fs::InteractiveFixedEffectsSolutionT{Tb, Tid, Ttime}
    scalefs::InteractiveFixedEffectsSolutionT{sTb, sTid, sTtime}
end

function size(fg::InteractiveFixedEffectsGradientT, i::Integer)
    if i == 1
        length(fg.fp.y)
    elseif i == 2
      length(fg.fs.b) + length(fg.fs.idpool) + length(fg.fs.timepool)
    end
end

Base.rank{Rank}(f::AbstractFactorGradient{Rank}) = Rank
size(fg::AbstractFactorGradient) = (size(fg, 1), size(fg, 2))
eltype(fg::AbstractFactorGradient) = Float64
colsumabs2!(fs::AbstractFactorSolution, fg::AbstractFactorGradient) = copy!(fs, fg.scalefs)

##############################################################################
##
## A_mul_B!
##
##############################################################################

@generated function A_mul_B!{Rank}(α::Number, fg::AbstractFactorGradient, fs::AbstractFactorSolution{Rank}, β::Number, y::AbstractVector{Float64})
    quote
        mα = convert(Float64, -α)
        A_mul_B!_X(mα, fg.fp, fs, β, y)
        @fastmath @inbounds @simd for i in 1:length(y)
            timei = fg.fp.timerefs[i]
            idi = fg.fp.idrefs[i]
            out = 0.0
            $(A_mul_B!_expr(Rank))
            y[i] += mα * fg.fp.sqrtw[i] * out
        end
        return y
    end
end

function A_mul_B!_expr(Rank)
    if Rank == 1
        quote
            out += (fg.fs.idpool[idi] * fs.timepool[timei] 
                + fg.fs.timepool[timei] * fs.idpool[idi])
        end
    else
        quote
            @nexprs $Rank r -> begin
                out += (fg.fs.idpool[r, idi] * fs.timepool[r, timei] 
                         + fg.fs.timepool[r, timei] * fs.idpool[r, idi])
            end
        end
    end
end

function A_mul_B!_X(α::Number, fp::InteractiveFixedEffectsModel, fs::InteractiveFixedEffectsSolutionT, β::Number, y::AbstractVector{Float64})
    Base.BLAS.gemm!('N', 'N', α, fp.X, fs.b, convert(Float64, β), y)
end
A_mul_B!_X(::Number, ::FactorModel, ::FactorSolution, β::Number, y::AbstractVector{Float64}) = safe_scale!(y, β)

##############################################################################
##
## Ac_mul_B!
##
##############################################################################

@generated function Ac_mul_B!{Rank}(α::Number, fg::AbstractFactorGradient, y::AbstractVector{Float64}, β::Number, fs::AbstractFactorSolution{Rank})
    quote
        mα = convert(Float64, -α)
        Ac_mul_B!_X(mα, fg.fp, y, β, fs)
        @inbounds @simd for i in 1:length(y)
            sqrtwi = mα * y[i] * fg.fp.sqrtw[i]
            idi = fg.fp.idrefs[i]
            timei = fg.fp.timerefs[i]
            $(Ac_mul_B!_expr(Rank))
        end
        return fs
    end
end   

function Ac_mul_B!_expr(Rank)
    if Rank == 1
        quote
           fs.idpool[idi] += sqrtwi * fg.fs.timepool[timei]
           fs.timepool[timei] += sqrtwi * fg.fs.idpool[idi]
        end
    else
        quote
            @nexprs $Rank r -> begin
                fs.idpool[r, idi] += sqrtwi * fg.fs.timepool[r, timei]
                fs.timepool[r, timei] += sqrtwi * fg.fs.idpool[r, idi]
            end
        end
    end
end

function Ac_mul_B!_X(α::Number, fp::InteractiveFixedEffectsModel, y::AbstractVector{Float64}, β::Number, fs::InteractiveFixedEffectsSolutionT)
    safe_scale!(fs, β)
    @inbounds @simd for k in 1:length(fs.b)
        out = zero(Float64)
        for i in 1:length(y)
           out += y[i] * fp.X[i, k]
        end
        fs.b[k] += α * out
    end
end
Ac_mul_B!_X(::Number, ::FactorModel, ::AbstractVector{Float64}, β::Number, fs::FactorSolution) = safe_scale!(fs, β)


##############################################################################
##
## f
##
##############################################################################

@generated function call{Rank}(fp::AbstractFactorModel{Rank}, 
    fs::AbstractFactorSolution{Rank}, out::Vector{Float64})
    quote
        copy!(out, fp.y)
        A_mul_B!_X(-1.0, fp, fs, 1.0, out)
        @fastmath @inbounds @simd for i in 1:length(out)
            sqrtwi = fp.sqrtw[i]
            idi = fp.idrefs[i]
            timei = fp.timerefs[i]
            $(f_expr!(Rank))
        end
        return out
    end
end

function f_expr!(Rank::Int)
    if Rank == 1
        quote
            out[i] -= sqrtwi * fs.idpool[idi] * fs.timepool[timei]
        end
    else
        quote
           @nexprs $Rank r -> begin
               out[i] -= sqrtwi * fs.idpool[r, idi] * fs.timepool[r, timei]
           end
        end
    end
end

##############################################################################
##
## g
##
##############################################################################

@generated function g!{Rank}(fs::AbstractFactorSolution{Rank}, fg::AbstractFactorGradient)
    quote
        copy!(fg.fs, fs)
        fill!(fg.scalefs.idpool, zero(Float64))
        fill!(fg.scalefs.timepool, zero(Float64))
        @inbounds @simd for i in 1:length(fg.fp.y)
            sqrtwi = fg.fp.sqrtw[i]
            idi = fg.fp.idrefs[i]
            timei = fg.fp.timerefs[i] 
            $(g_expr!(Rank))
        end
    end
end

function g_expr!(Rank::Int)
    if Rank == 1
        quote
            fg.scalefs.idpool[idi] += abs2(sqrtwi * fg.fs.timepool[timei])
            fg.scalefs.timepool[timei] += abs2(sqrtwi * fg.fs.idpool[idi])
        end
    else
        quote
            @nexprs $Rank r -> begin
                fg.scalefs.idpool[r, idi] += abs2(sqrtwi * fg.fs.timepool[r, timei])
                fg.scalefs.timepool[r, timei] += abs2(sqrtwi * fg.fs.idpool[r, idi])
            end
        end
    end
end



