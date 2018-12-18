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
import StatsModels: ModelMatrix, ModelFrame, Terms, coefnames, Formula, completecases, names!, @formula, evalcontrasts, check_non_redundancy!
import LeastSquaresOptim
using FillArrays
using Reexport
@reexport using FixedEffectModels
import FixedEffectModels: title, top


##############################################################################
##
## Exported methods and types 
##
##############################################################################

export InteractiveFixedEffectFormula,
InteractiveFixedEffectResult,
regife
##############################################################################
##
## Load files
##
##############################################################################
include("types.jl")

include("utils/models.jl")

include("methods/gauss_seidel.jl")
include("methods/ls.jl")

include("regife.jl")

end