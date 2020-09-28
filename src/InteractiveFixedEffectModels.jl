module InteractiveFixedEffectModels
##############################################################################
##
## Dependencies
##
##############################################################################
using DataFrames
using Distributions
using FillArrays
using FixedEffects
using LeastSquaresOptim
using LinearAlgebra
using Printf
using Statistics
using StatsBase
using StatsModels
using Tables
using Vcov

using Reexport
@reexport using FixedEffectModels

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