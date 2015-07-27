module PanelFactorModels

##############################################################################
##
## Dependencies
##
##############################################################################
using Compat
import DataArrays: RefArray, PooledDataVector, DataVector, PooledDataArray, DataArray
import DataFrames: DataFrame, AbstractDataFrame, ModelMatrix, ModelFrame, Terms, coefnames, Formula, complete_cases, names!
import FixedEffectModels: reg, demean!, getfe, decompose!, allvars, AbstractFixedEffect, FixedEffect, FixedEffectIntercept, FixedEffectSlope
import StatsBase: model_response, fit
import Optim: optimize, DifferentiableFunction, TwiceDifferentiableFunction
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
include("fitdataframe.jl")
end