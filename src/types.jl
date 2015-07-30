# Object constructed by the user
type SparseFactorModel
    id::Symbol
    time::Symbol
    rank::Int64
end

# type used internally to store idrefs, timerefs, factors, loadings
type PooledFactor{R}
    refs::Vector{R}
    pool::Matrix{Float64}
    old1pool::Matrix{Float64}
    old2pool::Matrix{Float64}
    storage1::Vector{Float64}
    storage2::Vector{Float64}
end
function PooledFactor{R}(refs::Vector{R}, l::Integer, rank::Integer)
    ans = fill(zero(Float64), l)
    PooledFactor(refs, fill(0.1, l, rank), fill(0.1, l, rank), fill(0.1, l, rank), ans, deepcopy(ans))
end


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

predict(x::RegressionFactorResult, df::AbstractDataFrame) = error("predict is not defined for linear factor models. Use the option save = true")
residuals(x::RegressionFactorResult, df::AbstractDataFrame) = error("residuals is not defined for linear factor models. Use the option save = true")
title(x::RegressionFactorResult) = "Linear Factor Model"
top(x::RegressionFactorResult) = [
            "Number of obs" sprint(showcompact, nobs(x));
            "Degree of freedom" sprint(showcompact, nobs(x) - df_residual(x));
            "R2" format_scientific(x.r2);
            "R2 within" format_scientific(x.r2_within);
            "Iterations" sprint(showcompact, x.iterations);
            "Converged" sprint(showcompact, x.converged)]
            


##############################################################################
##
## light weight type
## 
##############################################################################

# http://stackoverflow.com/a/30968709/3662288
type Ones <: AbstractVector{Float64}
    length::Int
end
Base.size(O::Ones) = O.length
Base.getindex(O::Ones, I::Int...) = one(Float64)
Base.broadcast!(op::Function, X::Matrix{Float64}, Y::Matrix{Float64}, O::Ones) = nothing
Base.broadcast!(op::Function, X::Vector{Float64}, Y::Vector{Float64}, O::Ones) = nothing
Base.scale!(X::Vector{Float64}, O::Ones) = nothing


function get_weight(df::AbstractDataFrame, weight::Symbol)
    w = convert(Vector{Float64}, df[weight])
    sqrtw = sqrt(w)
end
get_weight(df::AbstractDataFrame, weight::Nothing) = Ones(size(df, 1))