
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
    fp = FactorProblem(y, sqrtw, X, idf.refs, timef.refs, size(idf.pool, 2))
    fv = FactorVector(b, idf.pool, timef.pool)
    fg = FactorGradient(b, idf.pool, timef.pool, similar(b), idf.old1pool, timef.old1pool, fp)
    iterations, converged = dogleg!(fv, fg, similar(y), 
                                    (x, out) -> f!(x, out, fp), 
                                    g!)
    # rescale factors and loadings so that factors' * factors = Id
    copy!(idf.old1pool, fv.idpool)
    copy!(timef.old1pool, fv.timepool)
    rescale!(idf.pool, timef.pool, idf.old1pool, timef.old1pool)
    return fv.b, [iterations], [converged]
end

##############################################################################
##
## Factor Vector
##
##############################################################################

type FactorProblem{W, Rid, Rtime}
    y::Vector{Float64}
    sqrtw::W
    X::Matrix{Float64}
    idrefs::Vector{Rid}
    timerefs::Vector{Rtime}
    rank::Int
end

type FactorVector <: AbstractVector{Float64}
    b::Vector{Float64}
    idpool::Matrix{Float64}
    timepool::Matrix{Float64}
end

function length(x::FactorVector) 
    length(x.b) + length(x.idpool) + length(x.timepool)
end

function copy!(fv2::FactorVector, fv1::FactorVector)
    copy!(fv2.b, fv1.b)
    copy!(fv2.idpool, fv1.idpool)
    copy!(fv2.timepool, fv1.timepool)
    return fv2
end

function axpy!(α::Float64, fv1::FactorVector, fv2::FactorVector)
    axpy!(α, fv1.b, fv2.b)
    axpy!(α, fv1.idpool, fv2.idpool)
    axpy!(α, fv1.timepool, fv2.timepool)
    return fv2
end

function broadcast!(x, out::FactorVector, fv1::FactorVector, fv2::FactorVector)
    broadcast!(x, out.b, fv1.b, fv2.b)
    broadcast!(x, out.idpool, fv1.idpool, fv2.idpool)
    broadcast!(x, out.timepool, fv1.timepool, fv2.timepool)
end

function scale!(fv2::FactorVector, fv1::FactorVector, α::Float64)
    scale!(fv2.b, fv1.b, α)
    scale!(fv2.idpool, fv1.idpool, α)
    scale!(fv2.timepool, fv1.timepool, α)
    return fv2
end

function scale!(fv::FactorVector, α::Float64)
    scale!(fv, fv, α::Float64)
end

function dot(fv1::FactorVector, fv2::FactorVector)  
    out = (dot(fv1.b, fv2.b) 
        + dot(vec(fv1.idpool), vec(fv2.idpool)) 
        + dot(vec(fv1.timepool), vec(fv2.timepool))
        )
end

sumabs2(fv::FactorVector) = sumabs2(fv.b) + sumabs2(fv.idpool) + sumabs2(fv.timepool)
norm(fv::FactorVector) = sqrt(sumabs2(fv))
maxabs(fv::FactorVector) = max(maxabs(fv.b), maxabs(fv.idpool), maxabs(fv.timepool))


function fill!(fv::FactorVector, x)
    fill!(fv.b, x)
    fill!(fv.idpool, x)
    fill!(fv.timepool, x)
end

function similar(fv::FactorVector)
    FactorVector(similar(fv.b), similar(fv.idpool), similar(fv.timepool))
end


##############################################################################
##
## Factor Gradient
##
##############################################################################

#type created to fit in cgls
type FactorGradient{W, Rid, Rtime} <: AbstractMatrix{Float64}
    b::Vector{Float64}
    idpool::Matrix{Float64}
    timepool::Matrix{Float64}
    scaleb::Vector{Float64}
    scaleid::Matrix{Float64}
    scaletime::Matrix{Float64}
    fp::FactorProblem{W, Rid, Rtime}
end

function Ac_mul_B!(fv::FactorVector, fg::FactorGradient, y::AbstractVector{Float64})
    fill!(fv.b, zero(Float64))
    for k in 1:length(fv.b)
        @inbounds @simd for i in 1:length(y)
            fv.b[k] -= y[i] * fg.fp.X[i, k]
        end
    end
    fill!(fv.idpool, zero(Float64))
    for r in 1:fg.fp.rank
        @inbounds @simd for i in 1:length(y)
            fv.idpool[fg.fp.idrefs[i], r] -= y[i] * fg.fp.sqrtw[i] * fg.timepool[fg.fp.timerefs[i], r]
        end
    end
    fill!(fv.timepool, zero(Float64))
    for r in 1:fg.fp.rank
        @inbounds @simd for i in 1:length(y)
          fv.timepool[fg.fp.timerefs[i], r] -= y[i] * fg.fp.sqrtw[i] * fg.idpool[fg.fp.idrefs[i], r]
      end
    end
    return fv
end


function A_mul_B!(y::AbstractVector{Float64}, fg::FactorGradient, fv::FactorVector)
    fill!(y, zero(Float64))
    for k in 1:length(fv.b)
        @inbounds @simd for i in 1:length(y)
            y[i] -= fg.fp.X[i, k]  * fv.b[k]
        end
    end
    for r in 1:fg.fp.rank
        @inbounds @simd for i in 1:length(y)
            y[i] -= fg.fp.sqrtw[i] * fg.idpool[fg.fp.idrefs[i], r] * fv.timepool[fg.fp.timerefs[i], r] 
        end
    end
    for r in 1:fg.fp.rank
        @inbounds @simd for i in 1:length(y)
            y[i] -= fg.fp.sqrtw[i] * fg.timepool[fg.fp.timerefs[i], r] * fv.idpool[fg.fp.idrefs[i], r] 
        end
    end
    return y
end


function sumabs2!(fv::FactorVector, fg::FactorGradient) 
    copy!(fv.b, fg.scaleb)
    copy!(fv.idpool, fg.scaleid)
    copy!(fv.timepool, fg.scaletime)
end

##############################################################################
##
## Functions and Gradient
##
##############################################################################

function f!(fv::FactorVector, out::Vector{Float64}, fp::FactorProblem)
    copy!(out, fp.y)
    for k in 1:length(fv.b)
        @inbounds @simd for i in 1:length(out)
            out[i] -= fv.b[k] * fp.X[i, k] 
        end
    end
    for r in 1:fp.rank
        @inbounds @simd for i in 1:length(out)
            out[i] -= fp.sqrtw[i] * fv.idpool[fp.idrefs[i], r] * fv.timepool[fp.timerefs[i], r]
        end
    end
    return out
end


function g!(fv::FactorVector, fg::FactorGradient)
    copy!(fg.b, fv.b)
    copy!(fg.idpool, fv.idpool)
    copy!(fg.timepool, fv.timepool)
    fill!(fg.scaleb, zero(Float64))

    # fill scale
    for k in 1:length(fv.b)
        @inbounds @simd for i in 1:length(fg.fp.y)
            fg.scaleb[k] += abs2(fg.fp.X[i, k])
        end
    end
    fill!(fg.scaleid, zero(Float64))
    for r in 1:fg.fp.rank
        @inbounds @simd for i in 1:length(fg.fp.y)
            fg.scaleid[fg.fp.idrefs[i], r] += abs2(fg.fp.sqrtw[i] * fg.timepool[fg.fp.timerefs[i], r])
        end
    end
    fill!(fg.scaletime, zero(Float64))
    for r in 1:fg.fp.rank
        @inbounds @simd for i in 1:length(fg.fp.y)
            fg.scaletime[fg.fp.timerefs[i], r] += abs2(fg.fp.sqrtw[i] * fg.idpool[fg.fp.idrefs[i], r])
        end
    end
end



