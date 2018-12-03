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
    fp = FactorModel(copy(fp.y), fp.sqrtw, fp.idrefs, fp.timerefs, 1)
    N = size(fs.idpool, 1)
    T = size(fs.timepool, 1)
    fg = FactorGradient(fp,
                        FactorSolution(Array{Float64}(undef,N), Array{Float64}(undef, T)),
                        FactorSolution(Array{Float64}(undef,N), Array{Float64}(undef, T))
                       )
    nls = LeastSquaresOptim.LeastSquaresProblem(
                view(fs, :, 1), 
                similar(fp.y), 
                (y, x) -> f!(y, x, fp), 
                fg, 
                g!)
    if t == Val{:levenberg_marquardt}
        optimizer = LeastSquaresOptim.LevenbergMarquardt(LeastSquaresOptim.LSMR())
    else
        optimizer = LeastSquaresOptim.Dogleg(LeastSquaresOptim.LSMR())
    end

    full = LeastSquaresOptim.LeastSquaresProblemAllocated(nls, optimizer)
    for r in 1:fullrank
        fsr = view(fs, :, r)
        full.x = fsr
        result = LeastSquaresOptim.optimize!(full,
            x_tol = 1e-32, g_tol = 1e-32, f_tol = tol,  iterations = maxiter)
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

function fit!(t::Union{Type{Val{:levenberg_marquardt}}, Type{Val{:dogleg}}},
                         fp::InteractiveFixedEffectsModel{Rank},
                         fs::InteractiveFixedEffectsSolution{Rank}; 
                         maxiter::Integer = 10_000,
                         tol::Real = 1e-9,
                         lambda::Number = 0.0) where {Rank}
    scaleb = vec(sum(abs2, fp.X, dims = 1))
    fsT = InteractiveFixedEffectsSolutionT(fs.b, fs.idpool', fs.timepool')
    fg = InteractiveFixedEffectsGradientT(fp, 
                similar(fsT),
                InteractiveFixedEffectsSolutionT(scaleb, similar(fsT.idpool), similar(fsT.timepool)))
    nls = LeastSquaresOptim.LeastSquaresProblem(fsT, similar(fp.y), (y, x) -> f!(y, x, fp), fg, g!)
    if t == Val{:levenberg_marquardt}
        optimizer = LeastSquaresOptim.LevenbergMarquardt(LeastSquaresOptim.LSMR())
    else
        optimizer = LeastSquaresOptim.Dogleg(LeastSquaresOptim.LSMR())
    end
    full = LeastSquaresOptim.LeastSquaresProblemAllocated(nls, optimizer)
    temp = similar(fp.y)
    result = LeastSquaresOptim.optimize!(full;
            x_tol = 1e-32, g_tol = 1e-32, f_tol = tol,  iterations = maxiter)
    # rescale factors and loadings so that factors' * factors = Id
    fs = InteractiveFixedEffectsSolution(fsT.b, fsT.idpool', fsT.timepool')
    return rescale(fs), result.mul_calls, result.converged
end


##############################################################################
##
## Methods used in optimize! in LeastSquares
##
##############################################################################

function dot(fs1::AbstractArray{T}, fs2::AbstractArray{T})   where {T}
    out = zero(typeof(one(T) * one(T)))
    @inbounds @simd for i in eachindex(fs1)
        out += fs1[i] * fs2[i]
    end
    return out
end

function dot(x::AbstractArray{T}, y::AbstractArray{T}, w::AbstractArray{T}) where {T}
    out = zero(typeof(one(T) * one(T)))
    @inbounds @simd for i in eachindex(x)
        out += w[i] * x[i] * y[i]
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

    vars = [:(rmul!(x.$field, α)) for field in fieldnames(t)]
    expr = Expr(:block, vars...)
    @eval begin
        function rmul!(x::$t, α::Number)
            $expr
            return x
        end
    end

    vars = [:(clamp!(x.$field, α, β)) for field in fieldnames(t)]
    expr = Expr(:block, vars...)
    @eval begin
        function clamp!(x::$t, α::Number, β::Number)
            $expr
            return x
        end
    end


    vars = [:(copyto!(x1.$field, x2.$field)) for field in fieldnames(t)]
    expr = Expr(:block, vars...)
    @eval begin
        function copyto!(x1::$t, x2::$t)
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

    vars = [:(dot(x1.$field, x2.$field, w.$field)) for field in fieldnames(t)]
    expr = Expr(:call, :+, vars...)
    @eval begin
        function dot(x1::$t, x2::$t, w::$t)
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

    vars = [:(x.$field) for field in fieldnames(t)]
    expr = Expr(:call, Iterators.flatten, Expr(:call, tuple, vars...))
    @eval begin
        iterate(x::$t) = iterate($expr)
        iterate(x::$t, state) = iterate($expr, state)
    end
end
norm(x::AbstractFactorSolution) = sqrt(sum(abs2, x))

eltype(fg::AbstractFactorSolution) = Float64

function safe_rmul!(x, β)
    if β != 1
        β == 0 ? fill!(x, zero(eltype(x))) : rmul!(x, β)
    end
end



##############################################################################
##
## Factor Gradient
##
##############################################################################

abstract type AbstractFactorGradient{Rank} end


struct FactorGradient{Rank, W, Rid, Rtime, Tid, Ttime, sTid, sTtime} <: AbstractFactorGradient{Rank}
    fp::FactorModel{Rank, W, Rid, Rtime}
    fs::FactorSolution{Rank, Tid, Ttime}
    scalefs::FactorSolution{Rank, sTid, sTtime}
end

function size(fg::FactorGradient, i::Integer)
    if i == 1
        length(fg.fp.y)
    elseif i == 2
      length(fg.fs.idpool) + length(fg.fs.timepool)
    end
end

struct InteractiveFixedEffectsGradientT{Rank, W, Rid, Rtime, Tb, Tid, Ttime, sTb, sTid, sTtime} <: AbstractFactorGradient{Rank}
    fp::InteractiveFixedEffectsModel{Rank, W, Rid, Rtime}
    fs::InteractiveFixedEffectsSolutionT{Rank, Tb, Tid, Ttime}
    scalefs::InteractiveFixedEffectsSolutionT{Rank, sTb, sTid, sTtime}
end

function size(fg::InteractiveFixedEffectsGradientT, i::Integer)
    if i == 1
        length(fg.fp.y)
    elseif i == 2
      length(fg.fs.b) + length(fg.fs.idpool) + length(fg.fs.timepool)
    end
end

rank(f::AbstractFactorGradient{Rank}) where {Rank} = Rank 
size(fg::AbstractFactorGradient) = (size(fg, 1), size(fg, 2))
eltype(fg::AbstractFactorGradient) = Float64
adjoint(fg::AbstractFactorGradient) = Adjoint(fg)
LeastSquaresOptim.colsumabs2!(fs::AbstractFactorSolution, fg::AbstractFactorGradient) = copyto!(fs, fg.scalefs)

@generated function mul!(y::AbstractVector{Float64}, fg::AbstractFactorGradient{Rank}, fs::AbstractFactorSolution, α::Number, β::Number) where {Rank}
    if Rank == 1
        ex = quote
            out += (fg.fs.idpool[idi] * fs.timepool[timei] 
                + fg.fs.timepool[timei] * fs.idpool[idi])
        end
    else
        ex = quote
            @nexprs $Rank r -> begin
                out += (fg.fs.idpool[r, idi] * fs.timepool[r, timei] 
                         + fg.fs.timepool[r, timei] * fs.idpool[r, idi])
            end
        end
    end
    quote
        mα = convert(Float64, -α)
        mul!_X(y, fg.fp, fs, mα, β)
        @fastmath @inbounds @simd for i in 1:length(y)
            timei = fg.fp.timerefs[i]
            idi = fg.fp.idrefs[i]
            out = 0.0
            $ex
            y[i] += mα * fg.fp.sqrtw[i] * out
        end
        return y
    end
end

function mul!_X(y::AbstractVector{Float64}, fp::InteractiveFixedEffectsModel, fs::InteractiveFixedEffectsSolutionT, α::Number, β::Number)
    gemm!('N', 'N', α, fp.X, fs.b, convert(Float64, β), y)
end

function mul!_X(y::AbstractVector{Float64}, ::FactorModel, ::FactorSolution, ::Number, β::Number)
    safe_rmul!(y, β)
end

@generated function mul!(fs::AbstractFactorSolution{Rank}, Cfg::Adjoint{T, U}, y::AbstractVector{Float64}, α::Number, β::Number) where {Rank, T, U <: AbstractFactorGradient}
    if Rank == 1
        ex = quote
           fs.idpool[idi] += sqrtwi * fg.fs.timepool[timei]
           fs.timepool[timei] += sqrtwi * fg.fs.idpool[idi]
        end
    else
        ex = quote
            @nexprs $Rank r -> begin
                fs.idpool[r, idi] += sqrtwi * fg.fs.timepool[r, timei]
                fs.timepool[r, timei] += sqrtwi * fg.fs.idpool[r, idi]
            end
        end
    end
    quote
        fg = adjoint(Cfg)
        mα = convert(Float64, -α)
        mul!_X(fs, fg.fp, y, mα, β)
        @inbounds @simd for i in 1:length(y)
            sqrtwi = mα * y[i] * fg.fp.sqrtw[i]
            idi = fg.fp.idrefs[i]
            timei = fg.fp.timerefs[i]
            $ex
        end
        return fs
    end
end   

function mul!_X(fs::InteractiveFixedEffectsSolutionT, fp::InteractiveFixedEffectsModel, y::AbstractVector{Float64}, α::Number, β::Number)
    safe_rmul!(fs, β)
    @inbounds @simd for k in 1:length(fs.b)
        out = zero(Float64)
        for i in 1:length(y)
           out += y[i] * fp.X[i, k]
        end
        fs.b[k] += α * out
    end
end

function mul!_X(fs::FactorSolution, ::FactorModel, ::AbstractVector{Float64}, ::Number, β::Number, ) safe_rmul!(fs, β)
end


##############################################################################
##
## f! for LeastSquaresOptim
##
##############################################################################

@generated function f!(out::Vector{Float64}, fs::AbstractFactorSolution{Rank}, fp::AbstractFactorModel{Rank}) where {Rank}
    if Rank == 1
        ex = quote
            out[i] -= sqrtwi * fs.idpool[idi] * fs.timepool[timei]
        end
    else
        ex = quote
           @nexprs $Rank r -> begin
               out[i] -= sqrtwi * fs.idpool[r, idi] * fs.timepool[r, timei]
           end
        end
    end
    quote
        copyto!(out, fp.y)
        mul!_X(out, fp, fs, -1.0, 1.0)
        @fastmath @inbounds @simd for i in 1:length(out)
            sqrtwi = fp.sqrtw[i]
            idi = fp.idrefs[i]
            timei = fp.timerefs[i]
            $ex
        end
        return out
    end
end

##############################################################################
##
## g! for LeastSquaresOptim
##
##############################################################################

@generated function g!(fg::AbstractFactorGradient{Rank}, fs::AbstractFactorSolution{Rank}) where {Rank}
    if Rank == 1
        ex = quote
            fg.scalefs.idpool[idi] += abs2(sqrtwi * fg.fs.timepool[timei])
            fg.scalefs.timepool[timei] += abs2(sqrtwi * fg.fs.idpool[idi])
        end
    else
        ex = quote
            @nexprs $Rank r -> begin
                fg.scalefs.idpool[r, idi] += abs2(sqrtwi * fg.fs.timepool[r, timei])
                fg.scalefs.timepool[r, timei] += abs2(sqrtwi * fg.fs.idpool[r, idi])
            end
        end
    end
    quote
        copyto!(fg.fs, fs)
        fill!(fg.scalefs.idpool, zero(Float64))
        fill!(fg.scalefs.timepool, zero(Float64))
        @inbounds @simd for i in 1:length(fg.fp.y)
            sqrtwi = fg.fp.sqrtw[i]
            idi = fg.fp.idrefs[i]
            timei = fg.fp.timerefs[i] 
            $ex
        end
    end
end




