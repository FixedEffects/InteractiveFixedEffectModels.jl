module InteractiveFixedEffectModels
##############################################################################
##
## Dependencies
##
##############################################################################


import Base: length, copyto!, broadcast!, similar, map!, fill!, size, length, eltype,  convert, view, clamp!, adjoint, iterate
import LinearAlgebra: mul!, rmul!, Adjoint, rank, norm, dot, eigen!, axpy!, Symmetric, diagm, cholesky!
import LinearAlgebra.BLAS: gemm!
import Statistics: mean
import Printf: @sprintf
using Base.Cartesian

import StatsBase: coef, nobs, coeftable, vcov, predict, residuals, var, RegressionModel, model_response, stderror, confint, CoefTable,  dof_residual
import CategoricalArrays: CategoricalArray, CategoricalVector, compress, categorical, CategoricalPool, levels, droplevels!
import DataFrames: DataFrame, AbstractDataFrame, completecases, names!, ismissing
import LeastSquaresOptim
using FillArrays
using Distributions
using Reexport
import StatsModels: @formula,  FormulaTerm, Term, InteractionTerm, ConstantTerm, MatrixTerm, AbstractTerm, coefnames, columntable, missing_omit, termvars, schema, apply_schema, modelmatrix, response, terms, FunctionTerm
using FixedEffects
@reexport using FixedEffectModels

if !isdefined(FixedEffects, :AbstractFixedEffectSolver)
	AbstractFixedEffectSolver{T} = AbstractFixedEffectMatrix{T}
end

##############################################################################
##
## Exported methods and types 
##
##############################################################################

export InteractiveFixedEffectTerm,
InteractiveFixedEffectModel, 
ife,
regife
##############################################################################
##
## Load files
##
##############################################################################
include("types.jl")

include("methods/gauss_seidel.jl")
include("methods/ls.jl")

include("fit.jl")

end