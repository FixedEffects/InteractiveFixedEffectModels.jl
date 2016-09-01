module InteractiveFixedEffectModels

##############################################################################
##
## Dependencies
##
##############################################################################


import Distances: chebyshev
import Base: length, copy!, axpy!, broadcast!, scale!, dot, similar, Ac_mul_B!, A_mul_B!, sumabs2!, map!, sumabs2, maxabs, fill!, norm, maxabs, size, length, eltype, rank, convert, view, clamp!, dot, vecdot
using Base.Cartesian
import StatsBase: coef, nobs, coeftable, vcov, predict, residuals, var, RegressionModel, model_response, stderr, confint, fit, CoefTable,  df_residual
import DataArrays: RefArray, PooledDataVector, DataVector, PooledDataArray, DataArray
import DataFrames: DataFrame, AbstractDataFrame, ModelMatrix, ModelFrame, Terms, coefnames, Formula, complete_cases, names!, pool
import LeastSquaresOptim
using Reexport
@reexport using FixedEffectModels
import FixedEffectModels: title, top, Ones


##############################################################################
##
## Exported methods and types 
##
##############################################################################

export InteractiveFixedEffectModel,
InteractiveFixedEffectResult

##############################################################################
##
## Load files
##
##############################################################################
include("types.jl")

include("utils/models.jl")

include("methods/gauss_seidel.jl")
include("methods/ls.jl")

include("fit.jl")

end