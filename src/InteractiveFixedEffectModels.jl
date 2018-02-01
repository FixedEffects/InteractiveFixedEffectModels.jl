module InteractiveFixedEffectModels

##############################################################################
##
## Dependencies
##
##############################################################################


import Distances: chebyshev
import Base: length, copy!, axpy!, broadcast!, scale!, dot, similar, Ac_mul_B!, A_mul_B!, sumabs2!, map!, sumabs2, maxabs, fill!, norm, maxabs, size, length, eltype, rank, convert, view, clamp!, dot, vecdot, start, next, done
using Base.Cartesian
import StatsBase: coef, nobs, coeftable, vcov, predict, residuals, var, RegressionModel, model_response, stderr, confint, CoefTable,  df_residual
import Missings: Missing
import DataArrays: DataArray
import CategoricalArrays: CategoricalArray, CategoricalVector, compress, categorical, CategoricalPool, levels, droplevels!
import DataFrames: DataFrame, AbstractDataFrame, completecases, names!, ismissing
import StatsModels: ModelMatrix, ModelFrame, Terms, coefnames, Formula, completecases, names!,  @formula
import LeastSquaresOptim
using Reexport
@reexport using FixedEffectModels
import FixedEffectModels: title, top, Ones, reg
using Iterators


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