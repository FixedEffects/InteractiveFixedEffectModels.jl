module SparseFactorModels

##############################################################################
##
## Dependencies
##
##############################################################################

using Reexport
using Base.Cartesian
@reexport using FixedEffectModels
import FixedEffectModels: title, top
using Compat
import LeastSquares: optimize!, colsumabs2!, NonLinearLeastSquares,  NonLinearLeastSquaresProblem
import Base: length, copy!, axpy!, broadcast!, scale!, dot, similar, Ac_mul_B!, A_mul_B!, sumabs2!, map!, sumabs2, maxabs, fill!, norm, maxabs, size, length, eltype
import StatsBase: coef, nobs, coeftable, vcov, predict, residuals, var, RegressionModel, model_response, stderr, confint, fit, CoefTable,  df_residual
import DataArrays: RefArray, PooledDataVector, DataVector, PooledDataArray, DataArray
import DataFrames: DataFrame, AbstractDataFrame, ModelMatrix, ModelFrame, Terms, coefnames, Formula, complete_cases, names!
import Distances: sqeuclidean

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

include("ar.jl")

include("ls.jl")

include("fit.jl")

end