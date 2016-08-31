module InteractiveFixedEffectModels

##############################################################################
##
## Dependencies
##
##############################################################################

using Reexport
using Base.Cartesian
@reexport using FixedEffectModels
import FixedEffectModels: title, top, Ones
import Distances: chebyshev
using Compat
import LeastSquaresOptim
import Base: length, copy!, axpy!, broadcast!, scale!, dot, similar, Ac_mul_B!, A_mul_B!, sumabs2!, map!, sumabs2, maxabs, fill!, norm, maxabs, size, length, eltype, rank, convert, view, clamp!, dot, vecdot
import StatsBase: coef, nobs, coeftable, vcov, predict, residuals, var, RegressionModel, model_response, stderr, confint, fit, CoefTable,  df_residual
import DataArrays: RefArray, PooledDataVector, DataVector, PooledDataArray, DataArray
import DataFrames: DataFrame, AbstractDataFrame, ModelMatrix, ModelFrame, Terms, coefnames, Formula, complete_cases, names!, pool



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