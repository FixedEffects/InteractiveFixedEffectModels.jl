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
                         fp::FactorProblem,
                         fs::FactorSolution{Void}; 
                         maxiter::Integer = 100_000,
                         tol::Real = 1e-9,
                         lambda::Real = 0.0)
    # initialize
    iter = 0
    res = deepcopy(fp.y)
    fp = FactorProblem(res, fp.sqrtw, fp.X, fp.idrefs, fp.timerefs, rank(fp))
    idscale = Array(Float64, size(fs.idpool, 1))
    timescale = Array(Float64, size(fs.timepool, 1))
    idpool = Array(Float64, size(fs.idpool, 1))
    timepool = Array(Float64, size(fs.timepool, 1))
    iterationsv = Int[]
    convergedv = Bool[]
    idpoolr = slice(fs.idpool, :, 1)
    timepoolr = slice(fs.timepool, :, 1)
    fstmp = FactorSolution(idpoolr, timepoolr)
    fg = FactorGradient(idpool, timepool, 
                        idscale, timescale, fp)
    nls = NonLinearLeastSquares(fstmp, similar(fp.y), fp, fg, g!)

    full = NonLinearLeastSquaresProblem(nls, t)
    for r in 1:rank(fp)
        idpoolr = slice(fs.idpool, :, r)
        timepoolr = slice(fs.timepool, :, r)
        full.nls.x = FactorSolution(idpoolr, timepoolr)
        result = optimize!(full,
            xtol = 1e-32, grtol = 1e-32, ftol = tol,  iterations = maxiter)
        push!(iterationsv, result.mul_calls)
        push!(convergedv, result.converged)
        if r < rank(fp)
            rescale!(idpoolr, timepoolr)
            subtract_factor!(fp.y, fp.sqrtw, fp.idrefs, fp.timerefs, idpoolr, timepoolr)
        end
    end
    # rescale factors and loadings so that factors' * factors = Id
    fs.idpool, fs.timepool = rescale(fs.idpool, fs.timepool)
    return fs, maximum(iterationsv), all(convergedv)
end


##############################################################################
##
## Interactive FixedEffectModel
##
##############################################################################

function fit!(t::Union{Type{Val{:levenberg_marquardt}}, Type{Val{:dogleg}}},
                         fp::FactorProblem,
                         fs::FactorSolution; 
                         maxiter::Integer = 100_000,
                         tol::Real = 1e-9,
                         lambda::Real = 0.0)
    idpoolT = fs.idpool'
    timepoolT = fs.timepool'
    scaleb = vec(sumabs2(fp.X, 1))
    fs = FactorSolution(fs.b, idpoolT, timepoolT)
    fg = FactorGradient(similar(fs.b), similar(idpoolT), similar(timepoolT), scaleb, 
                        similar(idpoolT), similar(timepoolT), fp)
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
    fs.idpool, fs.timepool = rescale(fs.idpool', fs.timepool')
    return fs, result.mul_calls, result.converged
end

##############################################################################
##
## Factor Vector
##
##############################################################################
eltype(fg::FactorSolution) = Float64

function similar(fs::FactorSolution)
     return FactorSolution(similar(fs.b), similar(fs.idpool), similar(fs.timepool))
 end
 function similar(fs::FactorSolution{Void})
    return FactorSolution(similar(fs.idpool), similar(fs.timepool))
end

function length(fs::FactorSolution)
    return length(fs.b) + length(fs.idpool) + length(fs.timepool)
end
function length(fs::FactorSolution{Void})
    return length(fs.idpool) + length(fs.timepool)
end

function sumabs2(fs::FactorSolution)
    sumabs2(fs.b) + sumabs2(fs.idpool) + sumabs2(fs.timepool)
end
function sumabs2(fs::FactorSolution{Void})
    sumabs2(fs.idpool) + sumabs2(fs.timepool)
end

norm(fs::FactorSolution) = sqrt(sumabs2(fs))

function maxabs(fs::FactorSolution)
    max(maxabs(fs.b), maxabs(fs.idpool), maxabs(fs.timepool))
end
function maxabs(fs::FactorSolution{Void})
    max(maxabs(fs.idpool), maxabs(fs.timepool))
end

for t in (FactorSolution{Void}, FactorSolution)
    @eval begin
        function fill!(fs::$t, α::Number)
            $(t == FactorSolution ? :(fill!(fs.b, α)) : :nothing)
            fill!(fs.idpool, α)
            fill!(fs.timepool, α)
            return fs
        end

        function scale!(fs::$t, α::Number)
            $(t == FactorSolution ? :(scale!(fs.b, α)) : :nothing)
            scale!(fs.idpool, α)
            scale!(fs.timepool, α)
            return fs
        end


        function copy!(fs2::$t, fs1::$t)
            $(t == FactorSolution ? :(copy!(fs2.b, fs1.b)) : :nothing)
            copy!(fs2.idpool, fs1.idpool)
            copy!(fs2.timepool, fs1.timepool)
            return fs2
        end

        function axpy!(α::Number, fs1::$t, fs2::$t)
            $(t == FactorSolution ? :(axpy!(α, fs1.b, fs2.b)) : :nothing)
            axpy!(α, fs1.idpool, fs2.idpool)
            axpy!(α, fs1.timepool, fs2.timepool)
            return fs2
        end

        function map!(f, out::$t,  fs::$t...)
            $(t == FactorSolution ? :(map!(f, out.b, map(x -> x.b, fs)...)) : :nothing)
            map!(f, out.idpool, map(x -> x.idpool, fs)...)
            map!(f, out.timepool, map(x -> x.timepool, fs)...)
            return out
        end

        function dot(fs1::$t, fs2::$t)  
            out = $(t == FactorSolution ? :(dot(fs1.b, fs2.b)) : zero(Float64))
            @inbounds @simd for i in eachindex(fs1.idpool)
                out += fs1.idpool[i] * fs2.idpool[i]
            end
            @inbounds @simd for i in eachindex(fs1.timepool)
                out += fs1.timepool[i] * fs2.timepool[i]
            end
            return out
        end
    end
end

##############################################################################
##
## Factor Gradient
##
##############################################################################
type FactorGradient{Tb, Tid, Ttime, sTb, sTid, sTtime, W, TX, Rid, Rtime, Rank} 
    b::Tb
    idpool::Tid
    timepool::Ttime
    scaleb::sTb
    scaleid::sTid
    scaletime::sTtime
    fp::FactorProblem{W, TX, Rid, Rtime, Rank}
end

Base.rank{Tb, Tid, Ttime, sTb, sTid, sTtime, W, TX, Rid, Rtime, Rank}(f::FactorGradient{Tb, Tid, Ttime, sTb, sTid, sTtime, W, TX, Rid, Rtime, Rank}) = Rank

FactorGradient(idpool, timepool, scaleid, scaletime, fp) = FactorGradient(nothing, idpool, timepool, nothing, scaleid, scaletime, fp)

function size{Tb}(fg::FactorGradient{Tb}, i::Integer)
    if i == 1
        length(fg.fp.y)
    elseif i == 2
        if Tb != Void
            length(fg.b) + length(fg.idpool) + length(fg.timepool)
        else
            length(fg.idpool) + length(fg.timepool)
        end
    end
end
size(fg::FactorGradient) = (size(fg, 1), size(fg, 2))

eltype(fg::FactorGradient) = Float64

function Ac_mul_B!(α::Number, fg::FactorGradient{Void}, y::AbstractVector{Float64}, β::Number, fs::FactorSolution)
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
        fs.idpool[idi] += sqrtwi * fg.timepool[timei]
        fs.timepool[timei] += sqrtwi * fg.idpool[idi]
    end
    return fs
end

@generated function Ac_mul_B!{Tb, Tid, Ttime, sTb, sTid, sTtime, W, TX, Rid, Rtime, Rank}(α::Number, fg::FactorGradient{Tb, Tid, Ttime, sTb, sTid, sTtime, W, TX, Rid, Rtime, Rank}, y::AbstractVector{Float64}, β::Number, fs::FactorSolution)
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
                fs.idpool[r, idi] += sqrtwi * fg.timepool[r, timei]
                fs.timepool[r, timei] += sqrtwi * fg.idpool[r, idi]
            end
        end
        return fs
    end
end

function A_mul_B!(α::Number, fg::FactorGradient{Void}, fs::FactorSolution, β::Number, y::AbstractVector{Float64})
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
        out = (fg.idpool[idi] * fs.timepool[timei] 
                               + fg.timepool[timei] * fs.idpool[idi]
                               )
        y[i] += mα * fg.fp.sqrtw[i] * out
    end
    return y
end

@generated function A_mul_B!{Tb, Tid, Ttime, sTb, sTid, sTtime, W, TX, Rid, Rtime, Rank}(α::Number, fg::FactorGradient{Tb, Tid, Ttime, sTb, sTid, sTtime, W, TX, Rid, Rtime, Rank}, fs::FactorSolution, β::Number, y::AbstractVector{Float64})
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
                 out += (fg.idpool[r, idi] * fs.timepool[r, timei] 
                         + fg.timepool[r, timei] * fs.idpool[r, idi]
                         )
             end
            y[i] += mα * fg.fp.sqrtw[i] * out
        end
        return y
    end
end

for (t, x) in ((:(FactorSolution{Void}), nothing), 
                (:(FactorSolution), :(copy!(fs.b, fg.scaleb))))
    @eval begin
        function colsumabs2!(fs::$t, fg::FactorGradient) 
            $x
            copy!(fs.idpool, fg.scaleid)
            copy!(fs.timepool, fg.scaletime)
            return fs
        end
    end
end

##############################################################################
##
## Functions and Gradient for the function to minimize
##
##############################################################################

function call(fp::FactorProblem, fs::FactorSolution{Void}, out::Vector{Float64})
    copy!(out, fp.y)
    @fastmath @inbounds @simd for i in 1:length(out)
        sqrtwi = fp.sqrtw[i]
        idi = fp.idrefs[i]
        timei = fp.timerefs[i]
        out[i] -= sqrtwi * fs.idpool[idi] * fs.timepool[timei]
    end
    return out
end

@generated function call{W, TX, Rid, Rtime, Rank, Tb}(fp::FactorProblem{W, TX, Rid, Rtime, Rank}, fs::FactorSolution{Tb}, out::Vector{Float64})
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

function g!(fs::FactorSolution, fg::FactorGradient{Void})
    copy!(fg.idpool, fs.idpool)
    copy!(fg.timepool, fs.timepool)

    # fill scale
    fill!(fg.scaleid, zero(Float64))
    fill!(fg.scaletime, zero(Float64))
    @inbounds @simd for i in 1:length(fg.fp.y)
        sqrtwi = fg.fp.sqrtw[i]
        idi = fg.fp.idrefs[i]
        timei = fg.fp.timerefs[i] 
        fg.scaleid[idi] += abs2(sqrtwi * fg.timepool[timei])
        fg.scaletime[timei] += abs2(sqrtwi * fg.idpool[idi])
    end
end

@generated function g!{Tb, Tid, Ttime, sTb, sTid, sTtime, W, TX, Rid, Rtime, Rank}(fs::FactorSolution, fg::FactorGradient{Tb, Tid, Ttime, sTb, sTid, sTtime, W, TX, Rid, Rtime, Rank})
    quote
        copy!(fg.b, fs.b)
        copy!(fg.idpool, fs.idpool)
        copy!(fg.timepool, fs.timepool)

        # fill scale
        fill!(fg.scaleid, zero(Float64))
        fill!(fg.scaletime, zero(Float64))
        @inbounds @simd for i in 1:length(fg.fp.y)
            sqrtwi = fg.fp.sqrtw[i]
            idi = fg.fp.idrefs[i]
            timei = fg.fp.timerefs[i] 
            @nexprs $Rank r -> begin
                fg.scaleid[r, idi] += abs2(sqrtwi * fg.timepool[r, timei])
                fg.scaletime[r, timei] += abs2(sqrtwi * fg.idpool[r, idi])
            end
        end
    end
end
