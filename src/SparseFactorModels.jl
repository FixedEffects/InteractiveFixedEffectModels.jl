VERSION >= v"0.4.0-dev+6521" &&  __precompile__(true)

module SparseFactorModels

##############################################################################
##
## Dependencies
##
##############################################################################

using Compat
using Optim
import StatsBase: coef, nobs, coeftable, vcov, predict, residuals, var, RegressionModel, model_response, stderr, confint, fit, CoefTable,  df_residual
import DataArrays: RefArray, PooledDataVector, DataVector, PooledDataArray, DataArray
import DataFrames: DataFrame, AbstractDataFrame, ModelMatrix, ModelFrame, Terms, coefnames, Formula, complete_cases, names!
import FixedEffectModels: reg, demean!, getfe, decompose!, allvars, AbstractFixedEffect, FixedEffect, FixedEffectIntercept, FixedEffectSlope, VcovData, AbstractVcovMethod,AbstractVcovMethodData, VcovSimple, VcovWhite, VcovCluster, VcovMethodData, vcov!, AbstractRegressionResult, title, top
import Distances: sqeuclidean, chebyshev

##############################################################################
##
## Exported methods and types 
##
##############################################################################
export SparseFactorModel,
SparseFactorResult
##############################################################################
##
## Load files
##
##############################################################################
include("types.jl")
include("utils/models.jl")
include("utils/factors.jl")

include("algorithms/ar.jl")
include("algorithms/gd.jl")
include("algorithms/svd.jl")

include("fit.jl")

end