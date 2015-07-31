module SparseFactorModels

##############################################################################
##
## Dependencies
##
##############################################################################
using Compat
import StatsBase: coef, nobs, coeftable, vcov, predict, residuals, var, RegressionModel, model_response, stderr, confint, fit, CoefTable
import DataArrays: RefArray, PooledDataVector, DataVector, PooledDataArray, DataArray
import DataFrames: DataFrame, AbstractDataFrame, ModelMatrix, ModelFrame, Terms, coefnames, Formula, complete_cases, names!
import FixedEffectModels: reg, demean!, getfe, decompose!, allvars, AbstractFixedEffect, FixedEffect, FixedEffectIntercept, FixedEffectSlope, VcovData, AbstractVcovMethod,AbstractVcovMethodData, VcovSimple, VcovWhite, VcovCluster, VcovMethodData, vcov!, AbstractRegressionResult, title, top
import Optim: optimize, DifferentiableFunction, TwiceDifferentiableFunction
import GLM: df_residual
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
include("utils/others.jl")
include("utils/models.jl")
include("utils/factors.jl")

include("update!.jl")
include("fitfactors.jl")
include("fitolsfactors.jl")
include("fitdataframe.jl")
end