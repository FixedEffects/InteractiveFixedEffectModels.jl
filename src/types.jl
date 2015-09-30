##############################################################################
##
## Object constructed by the user
##
##############################################################################

# Object constructed by the user
type SparseFactorModel
    id::Symbol
    time::Symbol
    rank::Int64
end

abstract AbstractFactorModel
abstract AbstractFactorSolution

##############################################################################
##
## Factor Model
##
##############################################################################

type FactorModel{Rank, W, Rid, Rtime} <: AbstractFactorModel
    y::Vector{Float64}
    sqrtw::W
    idrefs::Vector{Rid}
    timerefs::Vector{Rtime}
end

function FactorModel{W, Rid, Rtime}(y::Vector{Float64}, sqrtw::W, idrefs::Vector{Rid}, timerefs::Vector{Rtime}, rank::Int)
    FactorModel{rank, W, Rid, Rtime}(y, sqrtw, idrefs, timerefs)
end

rank{Rank}(::FactorModel{Rank}) = Rank

type FactorSolution{Tid, Ttime} <: AbstractFactorSolution
    idpool::Tid
    timepool::Ttime
end

similar(fs::FactorSolution) = similar(idpool, timepool)

## subtract_factor! and subtract_b!
function subtract_factor!(y, fm::AbstractFactorModel, fs::AbstractFactorSolution)
    for r in 1:rank(fm)
        subtract_factor!(y, fm, FactorSolution(slice(fs.idpool, :, r), slice(fs.timepool, :, r)))
    end
end

function subtract_factor!{Tid <: AbstractVector, Ttime <: AbstractVector}(y, fm::AbstractFactorModel, fs::FactorSolution{Tid, Ttime})
    @inbounds @simd for i in 1:length(y)
        y[i] -= fm.sqrtw[i] * fs.idpool[fm.idrefs[i]] * fs.timepool[fm.timerefs[i]]
    end
end

## compute sum of squared residuals
function ssr(fm::FactorModel, fs::FactorSolution)
    out = zero(Float64)
    @inbounds @simd for i in 1:length(fm.y)
        out += abs2(fm.y[i] - fm.sqrtw[i] * fs.idpool[fm.idrefs[i]] * fs.timepool[fm.timerefs[i]])
    end 
    return out
end


## rescale a factor model
function reverse{R}(m::Matrix{R})
    out = similar(m)
    for j in 1:size(m, 2)
        invj = size(m, 2) + 1 - j 
        @inbounds @simd for i in 1:size(m, 1)
            out[i, j] = m[i, invj]
        end
    end
    return out
end
function rescale!{Tid <:AbstractVector{Float64}, Ttime <: AbstractVector{Float64}}(
    fs::FactorSolution{Tid, Ttime})
    out = norm(fs.timepool)
    @inbounds @simd for i in eachindex(fs.idpool)
        fs.idpool[i] *= out
    end
    invout = 1 / out
    @inbounds @simd for i in eachindex(fs.timepool)
        fs.timepool[i] *= invout
    end
end
# normalize factors and loadings so that F'F = Id, Lambda'Lambda diagonal
function rescale!(newfs::AbstractFactorSolution, fs::AbstractFactorSolution)
    if any(isnan, fs.timepool)
        copy!(newfs.timepool, fs.timepool)
        return newfs.idpool, newfs.timepool
    end
    U = eigfact!(Symmetric(At_mul_B(fs.timepool, fs.timepool)))
    sqrtDx = diagm(sqrt(abs(U[:values])))
    A_mul_B!(newfs.idpool,  fs.idpool,  U[:vectors] * sqrtDx)
    V = eigfact!(At_mul_B(newfs.idpool, newfs.idpool))
    A_mul_B!(newfs.idpool, fs.idpool, reverse(U[:vectors] * sqrtDx * V[:vectors]))
    A_mul_B!(newfs.timepool, fs.timepool, reverse(U[:vectors] * (sqrtDx \ V[:vectors])))
    return newfs
end

rescale(fs::FactorSolution) = rescale!(similar(fs), fs)


## Create dataframe from pooledfactors
function getfactors(fp::AbstractFactorModel, fs::AbstractFactorSolution)
    # partial out Y and X with respect to i.id x factors and i.time x loadings
    newfes = FixedEffect[]
    for r in 1:rank(fp)
        idinteraction = Array(Float64, length(fp.y))
        for i in 1:length(fp.y)
            idinteraction[i] = fs.timepool[fp.timerefs[i], r]
        end
        idfe = FixedEffect(fp.idrefs, size(fs.idpool, 1), fp.sqrtw, idinteraction, :id, :time, :(idxtime))
        push!(newfes, idfe)
        timeinteraction = Array(Float64, length(fp.y))
        for i in 1:length(fp.y)
            timeinteraction[i] = fs.idpool[fp.idrefs[i], r]
        end
        timefe = FixedEffect(fp.timerefs, size(fs.timepool, 1), fp.sqrtw, timeinteraction, :time, :id, :(timexid))
        push!(newfes, timefe)
    end
    # obtain the residuals and cross 
    return newfes
end



function DataFrame(fp::AbstractFactorModel, fs::AbstractFactorSolution, esample::BitVector)
    df = DataFrame()
    anyNA = all(esample)
    for r in 1:rank(fp)
        # loadings
        df[convert(Symbol, "loadings$r")] = build_column(fp.idrefs, fs.idpool[:, r], esample)
        df[convert(Symbol, "factors$r")] = build_column(fp.timerefs, fs.timepool[:, r], esample)
    end
    return df
end
function build_column(refs::Vector, pool::Vector, esample::BitVector)
    T = eltype(refs)
    newrefs = fill(zero(T), length(esample))
    newrefs[esample] = refs
    return PooledDataArray(RefArray(newrefs), pool)
end



##############################################################################
##
## Interactive Fixed Effect Models
##
##############################################################################

type InteractiveFixedEffectsModel{Rank, W, Rid, Rtime} <: AbstractFactorModel
    y::Vector{Float64}
    sqrtw::W
    X::Matrix{Float64}
    idrefs::Vector{Rid}
    timerefs::Vector{Rtime}
end

function InteractiveFixedEffectsModel{W, Rid, Rtime}(y::Vector{Float64}, sqrtw::W, X::Matrix{Float64}, idrefs::Vector{Rid}, timerefs::Vector{Rtime}, rank::Int)
    InteractiveFixedEffectsModel{rank, W, Rid, Rtime}(y, sqrtw, X, idrefs, timerefs)
end

rank{Rank}(::InteractiveFixedEffectsModel{Rank}) = Rank


type InteractiveFixedEffectsSolution{Tb, Tid, Ttime} <: AbstractFactorSolution
    b::Tb
    idpool::Tid
    timepool::Ttime
end

convert{Rank, W, Rid, Rtime}(::Type{FactorModel}, f::InteractiveFixedEffectsModel{Rank, W, Rid, Rtime}) = FactorModel{Rank, W, Rid, Rtime}(f.y, f.sqrtw, f.idrefs, f.timerefs)
convert(::Type{FactorSolution}, f::InteractiveFixedEffectsSolution) = FactorSolution(f.idpool, f.timepool)

function rescale(fs::InteractiveFixedEffectsSolution)
    fss = FactorSolution(fs.idpool, fs.timepool)
    newfss = similar(fss)
    rescale!(newfss, fss)
    InteractiveFixedEffectsSolution(fs.b, newfss.idpool, newfss.timepool)
end

##############################################################################
##
## Results
##
##############################################################################'

type FactorResult 
    esample::BitVector
    augmentdf::DataFrame

    ess::Float64
    iterations::Int64
    converged::Bool
end


# result
type InteractiveFixedEffectsResult <: AbstractRegressionResult
    coef::Vector{Float64}   # Vector of coefficients
    vcov::Matrix{Float64}   # Covariance matrix

    esample::BitVector      # Is the row of the original dataframe part of the estimation sample?
    augmentdf::DataFrame

    coefnames::Vector       # Name of coefficients
    yname::Symbol           # Name of dependent variable
    formula::Formula        # Original formula 

    nobs::Int64             # Number of observations
    df_residual::Int64      # degree of freedoms

    r2::Float64             # R squared
    r2_a::Float64           # R squared adjusted
    r2_within::Float64      # R within

    ess::Float64
    iterations::Int         # Number of iterations        
    converged::Bool         # Has the demeaning algorithm converged?

end

predict(::InteractiveFixedEffectsResult, ::AbstractDataFrame) = error("predict is not defined for linear factor models. Use the option save = true")
residuals(::InteractiveFixedEffectsResult, ::AbstractDataFrame) = error("residuals is not defined for linear factor models. Use the option save = true")
title(::InteractiveFixedEffectsResult) = "Linear Factor Model"
top(x::InteractiveFixedEffectsResult) = [
            "Number of obs" sprint(showcompact, nobs(x));
            "Degree of freedom" sprint(showcompact, nobs(x) - df_residual(x));
            "R2"  @sprintf("%.3f", x.r2);
            "R2 within"  @sprintf("%.3f", x.r2_within);
            "Iterations" sprint(showcompact, x.iterations);
            "Converged" sprint(showcompact, x.converged)
            ]




