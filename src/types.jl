# Object constructed by the user
type SparseFactorModel
    id::Symbol
    time::Symbol
    rank::Int64
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



