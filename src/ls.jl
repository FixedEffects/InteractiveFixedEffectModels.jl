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

@generated function similar{Tb}(fs::FactorSolution{Tb})
    quote
        if Tb != Void 
            return FactorSolution(similar(fs.b), similar(fs.idpool), similar(fs.timepool))
        else
            return FactorSolution(similar(fs.idpool), similar(fs.timepool))
        end
    end
end

@generated function length{Tb}(fs::FactorSolution{Tb})
    quote
        if Tb != Void 
            return length(fs.b) + length(fs.idpool) + length(fs.timepool)
        else
            return length(fs.idpool) + length(fs.timepool)
        end
    end
end
eltype(fg::FactorSolution) = Float64



@generated function fill!{Tb}(fs::FactorSolution{Tb}, α::Number)
    quote
        if Tb != Void
            fill!(fs.b, α)
        end
        fill!(fs.idpool, α)
        fill!(fs.timepool, α)
        return fs
    end
end

@generated function scale!{Tb}(fs1::FactorSolution{Tb}, α::Number)
    quote
        if Tb != Void
            scale!(fs1.b, α)
        end
        scale!(fs1.idpool, α)
        scale!(fs1.timepool, α)
        return fs1
    end
end

@generated function copy!{Tb}(fs2::FactorSolution{Tb}, fs1::FactorSolution{Tb})
    quote
        if Tb != Void
            copy!(fs2.b, fs1.b)
        end
        copy!(fs2.idpool, fs1.idpool)
        copy!(fs2.timepool, fs1.timepool)
        return fs2
    end
end

@generated function axpy!{Tb}(α::Number, fs1::FactorSolution{Tb}, fs2::FactorSolution{Tb})
    quote
        if Tb != Void
            axpy!(α, fs1.b, fs2.b)
        end
        axpy!(α, fs1.idpool, fs2.idpool)
        axpy!(α, fs1.timepool, fs2.timepool)
        return fs2
    end
end

@generated function map!{Tb}(f, out::FactorSolution{Tb},  fs::FactorSolution{Tb}...)
    quote
        if Tb != Void
            map!(f, out.b, map(x -> x.b, fs)...)
        end
        map!(f, out.idpool, map(x -> x.idpool, fs)...)
        map!(f, out.timepool, map(x -> x.timepool, fs)...)
        return out
    end
end


@generated function dot{Tb}(fs1::FactorSolution{Tb}, fs2::FactorSolution{Tb})  
    quote
        out = zero(Float64)
        if Tb != Void
            out = dot(fs1.b, fs2.b) 
        end
        @inbounds @simd for i in eachindex(fs1.idpool)
            out += fs1.idpool[i] * fs2.idpool[i]
        end
        @inbounds @simd for i in eachindex(fs1.timepool)
            out += fs1.timepool[i] * fs2.timepool[i]
        end
        return out
    end
end

@generated function sumabs2{Tb}(fs::FactorSolution{Tb})
    quote 
        out = zero(Float64)
        if Tb != Void 
            out += sumabs2(fs.b)
        end
        out += sumabs2(fs.idpool) + sumabs2(fs.timepool)
    end
end
norm(fs::FactorSolution) = sqrt(sumabs2(fs))
@generated function maxabs{Tb}(fs::FactorSolution{Tb})
    quote
        if Tb != Void 
            max(maxabs(fs.b), maxabs(fs.idpool), maxabs(fs.timepool))
        else
            max(maxabs(fs.idpool), maxabs(fs.timepool))
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

@generated function size{Tb, Tid, Ttime, sTb, sTid, sTtime, W, TX, Rid, Rtime, Rank}(
    fg::FactorGradient{Tb, Tid, Ttime, sTb, sTid, sTtime, W, TX, Rid, Rtime, Rank}, i::Integer)
    quote
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
end
size(fg::FactorGradient) = (size(fg, 1), size(fg, 2))
eltype(fg::FactorGradient) = Float64

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
        if Tb != Void
            for k in 1:length(fs.b)
                out = zero(Float64)
                @fastmath @inbounds @simd for i in 1:length(y)
                    out += y[i] * fg.fp.X[i, k]
                end
                fs.b[k] += mα * out
            end
        end
        @fastmath @inbounds @simd for i in 1:length(y)
            sqrtwi = mα * y[i] * fg.fp.sqrtw[i]
            idi = fg.fp.idrefs[i]
            timei = fg.fp.timerefs[i]
            if Tb != Void
                @nexprs $Rank r -> begin
                    fs.idpool[r, idi] += sqrtwi * fg.timepool[r, timei]
                    fs.timepool[r, timei] += sqrtwi * fg.idpool[r, idi]
                end
            else
                fs.idpool[idi] += sqrtwi * fg.timepool[timei]
                fs.timepool[timei] += sqrtwi * fg.idpool[idi]
            end
        end
        return fs
    end
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
        if Tb != Void
            Base.BLAS.gemm!('N', 'N', mα, fg.fp.X, fs.b, 1.0, y)
        end
        @fastmath @inbounds @simd for i in 1:length(y)
            timei = fg.fp.timerefs[i]
            idi = fg.fp.idrefs[i]
            if Tb != Void
                out = 0.0
                @nexprs $Rank r -> begin
                     out += (fg.idpool[r, idi] * fs.timepool[r, timei] 
                             + fg.timepool[r, timei] * fs.idpool[r, idi]
                             )
                 end
             else
                out = (fg.idpool[idi] * fs.timepool[timei] 
                        + fg.timepool[timei] * fs.idpool[idi]
                        )
            end
            y[i] += mα * fg.fp.sqrtw[i] * out
        end
        return y
    end
end

@generated function colsumabs2!{Tb}(fs::FactorSolution{Tb}, fg::FactorGradient) 
    quote
        if Tb != Void 
            copy!(fs.b, fg.scaleb)
        end
        copy!(fs.idpool, fg.scaleid)
        copy!(fs.timepool, fg.scaletime)
        return fs
    end
end

##############################################################################
##
## Functions and Gradient for the function to minimize
##
##############################################################################
@generated function call{W, TX, Rid, Rtime, Rank, Tb}(fp::FactorProblem{W, TX, Rid, Rtime, Rank}, fs::FactorSolution{Tb}, out::Vector{Float64})
    quote
        copy!(out, fp.y)
        if Tb != Void
            BLAS.gemm!('N', 'N', -1.0, fp.X, fs.b, 1.0, out)
        end
        @fastmath @inbounds @simd for i in 1:length(out)
            sqrtwi = fp.sqrtw[i]
            idi = fp.idrefs[i]
            timei = fp.timerefs[i]
            if Tb != Void
                @nexprs $Rank r -> begin
                    out[i] -= sqrtwi * fs.idpool[r, idi] * fs.timepool[r, timei]
                end
            else
                out[i] -= sqrtwi * fs.idpool[idi] * fs.timepool[timei]
            end
        end
        return out
    end
end

@generated function g!{Tb, Tid, Ttime, sTb, sTid, sTtime, W, TX, Rid, Rtime, Rank}(fs::FactorSolution, fg::FactorGradient{Tb, Tid, Ttime, sTb, sTid, sTtime, W, TX, Rid, Rtime, Rank})
    quote
        if Tb != Void
            copy!(fg.b, fs.b)
        end
        copy!(fg.idpool, fs.idpool)
        copy!(fg.timepool, fs.timepool)

        # fill scale
        fill!(fg.scaleid, zero(Float64))
        fill!(fg.scaletime, zero(Float64))
        @fastmath @inbounds @simd for i in 1:length(fg.fp.y)
            sqrtwi = fg.fp.sqrtw[i]
            idi = fg.fp.idrefs[i]
            timei = fg.fp.timerefs[i] 
            if Tb == Void
                fg.scaleid[idi] += abs2(sqrtwi * fg.timepool[timei])
                fg.scaletime[timei] += abs2(sqrtwi * fg.idpool[idi])
            else
                @nexprs $Rank r -> begin
                    fg.scaleid[r, idi] += abs2(sqrtwi * fg.timepool[r, timei])
                    fg.scaletime[r, timei] += abs2(sqrtwi * fg.idpool[r, idi])
                end
            end
        end
    end
end