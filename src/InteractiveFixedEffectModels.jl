module InteractiveFixedEffectModels
##############################################################################
##
## Dependencies
##
##############################################################################


using Base
using LinearAlgebra
using Statistics
using Printf

using StatsBase
using StatsModels
using CategoricalArrays
using DataFrames
using LeastSquaresOptim
using FillArrays
using Distributions
using Reexport
using FixedEffects
@reexport using FixedEffectModels


if !isdefined(FixedEffectModels, :ModelTerm)
    ModelTerm = Model
end
##############################################################################
##
## Exported methods and types 
##
##############################################################################

export InteractiveFixedEffectTerm,
InteractiveFixedEffectModel, 
ife,
regife
##############################################################################
##
## Load files
##
##############################################################################
include("types.jl")

include("methods/gauss_seidel.jl")
include("methods/ls.jl")

include("fit.jl")

end