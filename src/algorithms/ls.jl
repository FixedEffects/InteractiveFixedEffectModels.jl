##############################################################################
##
## Solution method: minimize by sparse least square optimization
##
##############################################################################

function fit!{W, Rid, Rtime}(t::Union{Type{Val{:lm}}, Type{Val{:dl}}}, 
                         fp::FactorProblem{W, Matrix{Float64}, Rid, Rtime},
                         fs::FactorSolution{Vector{Float64}}; 
                         maxiter::Integer = 100_000,
                         tol::Real = 1e-9,
                         lambda::Real = 0.0)
    scaleb = vec(sumabs2(fp.X, 1))
    fg = FactorGradient(fs.b, fs.idpool, fs.timepool, scaleb, 
                        similar(fs.idpool), similar(fs.timepool), fp)
    nls = NonLinearLeastSquares(fs, similar(fp.y), (x, out) -> f!(x, out, fp), fg, g!)
    if t == Val{:lm}
        result = optimize!(nls ; method = :levenberg_marquardt,
            xtol = tol, grtol = 1e-32, ftol = 1e-32,  iterations = maxiter)
    else
          result = optimize!(nls ;  method = :dogleg,
          xtol = tol, grtol = 1e-32, ftol = 1e-32,  iterations = maxiter)
    end
    # rescale factors and loadings so that factors' * factors = Id
    fs.idpool, fs.timepool = rescale(fs.idpool, fs.timepool)
    return fs, [result.mul_calls], [result.converged]
end

##############################################################################
##
## Factor Vector
##
##############################################################################

function similar(fs::FactorSolution{Vector{Float64}})
    return FactorSolution(similar(fs.b), similar(fs.idpool), similar(fs.timepool))
end

function fill!(fs::FactorSolution{Vector{Float64}}, α::Number)
    fill!(fs.b, α)
    fill!(fs.idpool, α)
    fill!(fs.timepool, α)
    return fs
end

function scale!(fs1::FactorSolution{Vector{Float64}}, α::Number)
    scale!(fs1.b, α)
    scale!(fs1.idpool, α)
    scale!(fs1.timepool, α)
    return fs1
end

function copy!(fs2::FactorSolution{Vector{Float64}}, fs1::FactorSolution{Vector{Float64}})
    copy!(fs2.b, fs1.b)
    copy!(fs2.idpool, fs1.idpool)
    copy!(fs2.timepool, fs1.timepool)
    return fs2
end

function axpy!(α::Number, fs1::FactorSolution{Vector{Float64}}, fs2::FactorSolution{Vector{Float64}})
    axpy!(α, fs1.b, fs2.b)
    axpy!(α, fs1.idpool, fs2.idpool)
    axpy!(α, fs1.timepool, fs2.timepool)
    return fs2
end

function map!(f, out::FactorSolution{Vector{Float64}},  fs::FactorSolution{Vector{Float64}}...)
    map!(f, out.b, map(x -> x.b, fs)...)
    map!(f, out.idpool, map(x -> x.idpool, fs)...)
    map!(f, out.timepool, map(x -> x.timepool, fs)...)
    return out
end

function dot(fs1::FactorSolution{Vector{Float64}}, fs2::FactorSolution{Vector{Float64}})  
    out = dot(fs1.b, fs2.b) 
    for i in 1:size(fs1.idpool, 2)
        out += dot(slice(fs1.idpool, :, i), slice(fs1.idpool, :, i)) 
    end
    for i in 1:size(fs2.timepool, 2)
        out += dot(slice(fs1.timepool, :, i), slice(fs1.timepool, :, i)) 
    end
    return out
end

sumabs2(fs::FactorSolution{Vector{Float64}}) = sumabs2(fs.b) + sumabs2(fs.idpool) + sumabs2(fs.timepool)
norm(fs::FactorSolution{Vector{Float64}}) = sqrt(sumabs2(fs))
maxabs(fs::FactorSolution{Vector{Float64}}) = max(maxabs(fs.b), maxabs(fs.idpool), maxabs(fs.timepool))

##############################################################################
##
## Factor Gradient
##
##############################################################################

#type created to fit in cgls
type FactorGradient{W, Rid, Rtime} 
    b::Vector{Float64}
    idpool::Matrix{Float64}
    timepool::Matrix{Float64}
    scaleb::Vector{Float64}
    scaleid::Matrix{Float64}
    scaletime::Matrix{Float64}
    fp::FactorProblem{W, Matrix{Float64}, Rid, Rtime}
end

function Ac_mul_B!(α::Number, fg::FactorGradient, y::AbstractVector{Float64}, β::Number, fs::FactorSolution{Vector{Float64}})
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
            out -= y[i] * fg.fp.X[i, k]
        end
        fs.b[k] += α * out
    end
    for r in 1:fg.fp.rank
        @inbounds @simd for i in 1:length(y)
            fs.idpool[fg.fp.idrefs[i], r] -= α * y[i] * fg.fp.sqrtw[i] * fg.timepool[fg.fp.timerefs[i], r]
        end
    end
    for r in 1:fg.fp.rank
        @inbounds @simd for i in 1:length(y)
          fs.timepool[fg.fp.timerefs[i], r] -= α * y[i] * fg.fp.sqrtw[i] * fg.idpool[fg.fp.idrefs[i], r]
      end
    end
    return fs
end

function Ac_mul_B!(fs::FactorSolution{Vector{Float64}}, fg::FactorGradient, y::AbstractVector{Float64})
    Ac_mul_B!(1, fg, y, 0, fs)
end

function A_mul_B!(α::Number, fg::FactorGradient, fs::FactorSolution{Vector{Float64}}, β::Number, y::AbstractVector{Float64})
    if β != 1.
        if β == 0.
            fill!(y, 0.)
        else
            scale!(y, β)
        end
    end
    Base.BLAS.gemm!('N', 'N', -convert(Float64, α), fg.fp.X, fs.b, 1.0, y)
    for r in 1:fg.fp.rank
        @inbounds @simd for i in 1:length(y)
            y[i] -= α * fg.fp.sqrtw[i] * fg.idpool[fg.fp.idrefs[i], r] * fs.timepool[fg.fp.timerefs[i], r] 
        end
    end
    for r in 1:fg.fp.rank
        @inbounds @simd for i in 1:length(y)
            y[i] -= α * fg.fp.sqrtw[i] * fg.timepool[fg.fp.timerefs[i], r] * fs.idpool[fg.fp.idrefs[i], r] 
        end
    end
    return y
end

function A_mul_B!(y::AbstractVector{Float64}, fg::FactorGradient, fs::FactorSolution{Vector{Float64}})
    A_mul_B!(1, fg, fs, 0, y)
end


function colsumabs2!(fs::FactorSolution{Vector{Float64}}, fg::FactorGradient) 
    copy!(fs.b, fg.scaleb)
    copy!(fs.idpool, fg.scaleid)
    copy!(fs.timepool, fg.scaletime)
    return fs
end

##############################################################################
##
## Functions and Gradient for the function to minimize
##
##############################################################################

function f!(fs::FactorSolution{Vector{Float64}}, out::Vector{Float64}, fp::FactorProblem)
    copy!(out, fp.y)
    BLAS.gemm!('N', 'N', -1.0, fp.X, fs.b, 1.0, out)
    for r in 1:fp.rank
        @inbounds @simd for i in 1:length(out)
            out[i] -= fp.sqrtw[i] * fs.idpool[fp.idrefs[i], r] * fs.timepool[fp.timerefs[i], r]
        end
    end
    return out
end

function g!(fs::FactorSolution{Vector{Float64}}, fg::FactorGradient)
    copy!(fg.b, fs.b)
    copy!(fg.idpool, fs.idpool)
    copy!(fg.timepool, fs.timepool)

    # fill scale
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