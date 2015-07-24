module PanelFactorModels

##############################################################################
##
## Dependencies
##
##############################################################################
using Compat
import DataArrays: RefArray, PooledDataVector, DataVector
import DataFrames: DataFrame, AbstractDataFrame, ModelMatrix, ModelFrame, Terms, coefnames, Formula, complete_cases, names!
using FixedEffectModels
import StatsBase: fit, model_response
import Optim: optimize, DifferentiableFunction, TwiceDifferentiableFunction
##############################################################################
##
## Exported methods and types 
##
##############################################################################
export PanelFactorModel,
PanelFactorResult,
PanelFactorModelResult
##############################################################################
##
## Load files
##
##############################################################################
include("utils.jl")
include("types.jl")
include("update!.jl")
include("fitvariable.jl")
include("fitmodel.jl")

end