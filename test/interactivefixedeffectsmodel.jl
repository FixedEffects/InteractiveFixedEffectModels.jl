using DataFrames, InteractiveFixedEffectModels, CSV, LinearAlgebra, Test

precision = 2e-1
df = CSV.read(joinpath(dirname(pathof(InteractiveFixedEffectModels)), "../dataset/Cigar.csv"))

df[:pState] = categorical(df[:State])
df[:pYear] = categorical(df[:Year])

for method in [:dogleg, :levenberg_marquardt, :gauss_seidel]
	println(method)
	model = @model Sales ~ Price ife = (pState + pYear, 1) method = $(method) save = true
	result = regife(df, model)
	@test norm(result.coef ./ [328.1653237715761, -1.0415042260420706] .- 1)  < precision
	@test norm(abs(result.augmentdf[1, :factors1]) / 0.228 - 1) < precision
	@test norm(abs(result.augmentdf[1, :loadings1]) / 851.514 - 1) < precision
	@test norm(result.augmentdf[1, :residuals] / -9.7870 - 1) < precision
	@test result.r2_within > 0.0

	show(result)
	model = @model Sales ~ Price ife = (pState + pYear, 2) method = $(method) save = true
	result = regife(df, model)
	@test norm(result.coef ./ [163.01350, -0.40610] .- 1) < precision
	@test norm(abs(result.augmentdf[1, :factors1]) / 0.227 - 1) < precision
	@test norm(abs(result.augmentdf[1, :loadings1]) / 184.1897 - 1) < precision
	@test norm(result.augmentdf[1, :residuals]  / -1.774 - 1) < precision
	@test result.r2_within > 0.0

	model = @model Sales ~ Price ife = (pState + pYear, 2) fe = pState method = $(method) save = true
	result = regife(df, model)
	@test norm(result.coef ./ -0.425389 .- 1) < precision
	@test norm(abs(result.augmentdf[1, :factors1])  / 0.2474 - 1) < precision
	@test norm(abs(result.augmentdf[1, :loadings1]) / 123.460 - 1) < precision
	@test norm(result.augmentdf[1, :residuals] / -5.222 - 1) < precision
	@test result.r2_within > 0.0

	model = @model Sales ~ Price ife = (pState + pYear, 2) fe = pYear method = $(method) save = true
	result = regife(df, model)
	@test norm(result.coef ./ -0.3744296120563005 .- 1) < precision
	@test norm(abs(result.augmentdf[1, :factors1])  / 0.1918 - 1) < precision
	@test norm(abs(result.augmentdf[1, :loadings1]) / 102.336 - 1) < precision
	@test norm(result.augmentdf[1, :residuals] / -2.2153 - 1) < precision
	@test result.r2_within > 0.0

	model = @model Sales ~ Price ife = (pState + pYear, 2) fe = pState + pYear method = $(method) save = true
	result = regife(df, model)
	@test norm(result.coef ./ -0.524157 .- 1) < precision
	@test norm(abs(result.augmentdf[1, :factors1]) / 0.256 - 1) < precision
	@test norm(abs(result.augmentdf[1, :loadings1]) / 60.0481 - 1) < precision
	@test norm(result.augmentdf[1, :residuals] / -5.614 - 1) < precision
	@test result.r2_within > 0.0

	# subset
	model = @model Sales ~ Price ife = (pState + pYear, 2) fe = pState method = $(method) subset = (State .<= 30) save = true
	result = regife(df, model)
	@test size(result.augmentdf, 1) == size(df, 1)
	@test norm(abs(result.augmentdf[:factors1][1]) / 0.25965 - 1) < precision
	@test norm(abs(result.augmentdf[:loadings1][1]) / 107.832 - 1) < precision
	@test norm(abs(result.augmentdf[:factors2][1])  / 0.2655  - 1) < precision
	@test norm(abs(result.augmentdf[:loadings2][1]) / 15.551 - 1) < precision
	@test norm(result.augmentdf[:residuals][1] / -3.8624  - 1) < precision
	@test norm(result.augmentdf[:pState][1] / 131.6162 - 1) < precision
end

# check high dimentional fixed effects are part of factor models
df[:pState2] = categorical(df[:State])
@test_throws ErrorException regife(df, @model(Sales ~ Price, ife = (pState + pYear, 2), fe = pState2, weights = Pop, method = levenberg_marquardt, save = true))


# local minima
using DataFrames, InteractiveFixedEffectModels, Test
precision = 2e-1


for method in [:levenberg_marquardt, :dogleg]
	println(method)

	df = CSV.read(joinpath(dirname(pathof(InteractiveFixedEffectModels)), "../dataset/Cigar.csv"))
	df[:pState] = categorical(df[:State])
	df[:pYear] = categorical(df[:Year])
	model = @model Sales ~ Price ife = (pState + pYear, 1) fe = pState weights = Pop method = $(method) save = true
	result = regife(df, model)
	model = @model Sales ~ Price ife = (pState + pYear, 2) fe = pState weights = Pop method = $(method) save = true
	result = regife(df, model)

	df = CSV.read(joinpath(dirname(pathof(InteractiveFixedEffectModels)), "../dataset/EmplUK.csv"))
	df[:id1] = df[:Firm]
	df[:id2] = df[:Year]
	df[:pid1] = categorical(df[:id1])
	df[:pid2] = categorical(df[:id2])
	df[:y] = df[:Wage]
	df[:x1] = df[:Emp]
	df[:w] = df[:Output]
	model = @model y ~ x1 ife = (pid1 + pid2, 2) method = $(method) save = true
	result = regife(df, model)
	@test norm(result.coef ./ [4.53965, -0.0160858] .- 1) < precision
	model = @model y ~ x1 ife = (pid1 + pid2, 2) weights = w method = $(method) save = true
	result = regife(df, model)
	@test norm(result.coef ./ [3.47551,-0.017366] .- 1) < precision
	model = @model y ~ x1 ife = (pid1 + pid2, 1) weights = w method = $(method) save = true
	result = regife(df, model)
	@test norm(result.coef ./ [ -2.62105, -0.0470005] .- 1) < precision
end





