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

##############################################################################
##
## Internally
##
##############################################################################

type FactorProblem{W, TX, Rid, Rtime}
    y::Vector{Float64}
    sqrtw::W
    X::TX
    idrefs::Vector{Rid}
    timerefs::Vector{Rtime}
    rank::Int
end
function FactorProblem(y::Vector{Float64}, sqrtw, idrefs::Vector, timerefs::Vector, rank::Int)
    FactorProblem(y, sqrtw, nothing, idrefs, timerefs, rank)
end

type FactorSolution{Tb}
    b::Tb
    idpool::Matrix{Float64}
    timepool::Matrix{Float64}
end
FactorSolution(idpool, timepool) = FactorSolution(nothing, idpool, timepool)


function getfactors(fp::FactorProblem,fs::FactorSolution)
    # partial out Y and X with respect to i.id x factors and i.time x loadings
    newfes = FixedEffect[]
    for r in 1:fp.rank
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

##############################################################################
##
## Result type
##
##############################################################################

# object returned when fitting variable
type SparseFactorResult 
    esample::BitVector
    augmentdf::DataFrame

    ess::Float64
    iterations::Vector{Int64}
    converged::Vector{Bool}
end

# object returned when fitting linear model
type RegressionFactorResult <: AbstractRegressionResult
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

predict(::RegressionFactorResult, ::AbstractDataFrame) = error("predict is not defined for linear factor models. Use the option save = true")
residuals(::RegressionFactorResult, ::AbstractDataFrame) = error("residuals is not defined for linear factor models. Use the option save = true")
title(::RegressionFactorResult) = "Linear Factor Model"
top(x::RegressionFactorResult) = [
            "Number of obs" sprint(showcompact, nobs(x));
            "Degree of freedom" sprint(showcompact, nobs(x) - df_residual(x));
            "R2"  @sprintf("%.3f", x.r2);
            "R2 within"  @sprintf("%.3f", x.r2_within);
            "Iterations" sprint(showcompact, x.iterations);
            "Converged" sprint(showcompact, x.converged)
            ]






