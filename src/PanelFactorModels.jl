module PanelFactorModels

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
export PanelFactorModel,
PanelFactorResult
##############################################################################
##
## Load files
##
##############################################################################
include("utils.jl")
include("update!.jl")
include("fitvariable.jl")
include("fitmodel.jl")
include("types.jl")
include("fitdataframe.jl")
end