using DataFrames, InteractiveFixedEffectModels, CSV, LinearAlgebra, Test

precision = 2e-1
df = CSV.read(joinpath(dirname(pathof(InteractiveFixedEffectModels)), "../dataset/Cigar.csv"))

df.pState = categorical(df.State)
df.pYear = categorical(df.Year)

for method in [:dogleg, :levenberg_marquardt, :gauss_seidel]
	println(method)
	model = @model Sales ~ Price ife = (pState + pYear, 1)
	result = regife(df, model, method = method, save = true)
	@test norm(result.coef ./ [328.1653237715761, -1.0415042260420706] .- 1)  < precision
	@test norm(abs(result.augmentdf[1, :factors1]) / 0.228 - 1) < precision
	@test norm(abs(result.augmentdf[1, :loadings1]) / 851.514 - 1) < precision
	@test norm(result.augmentdf[1, :residuals] / -9.7870 - 1) < precision
	@test result.r2_within > 0.0

	show(result)
	model = @model Sales ~ Price ife = (pState + pYear, 2)
	result = regife(df, model, method = method, save = true)
	@test norm(result.coef ./ [163.01350, -0.40610] .- 1) < precision
	@test norm(abs(result.augmentdf[1, :factors1]) / 0.227 - 1) < precision
	@test norm(abs(result.augmentdf[1, :loadings1]) / 184.1897 - 1) < precision
	@test norm(result.augmentdf[1, :residuals]  / -1.774 - 1) < precision
	@test result.r2_within > 0.0

	model = @model Sales ~ Price ife = (pState + pYear, 2) fe = pState
	result = regife(df, model, method = method, save = true)
	@test norm(result.coef ./ -0.425389 .- 1) < precision
	@test norm(abs(result.augmentdf[1, :factors1])  / 0.2474 - 1) < precision
	@test norm(abs(result.augmentdf[1, :loadings1]) / 123.460 - 1) < precision
	@test norm(result.augmentdf[1, :residuals] / -5.222 - 1) < precision
	@test result.r2_within > 0.0

	model = @model Sales ~ Price ife = (pState + pYear, 2) fe = pYear
	result = regife(df, model, method = method, save = true)
	@test norm(result.coef ./ -0.3744296120563005 .- 1) < precision
	@test norm(abs(result.augmentdf[1, :factors1])  / 0.1918 - 1) < precision
	@test norm(abs(result.augmentdf[1, :loadings1]) / 102.336 - 1) < precision
	@test norm(result.augmentdf[1, :residuals] / -2.2153 - 1) < precision
	@test result.r2_within > 0.0

	# not working since STatsModels 0.6. The issue is that  fs.timepool now has a zero eigenvalue
	#model = @model Sales ~ Price ife = (pState + pYear, 2) fe = pState + pYear
	#result = regife(df, model, method = method, save = true)
	#@test norm(result.coef ./ -0.524157 .- 1) < precision
	#@test norm(abs(result.augmentdf[1, :factors1]) / 0.256 - 1) < precision
	#@test norm(abs(result.augmentdf[1, :loadings1]) / 60.0481 - 1) < precision
	#@test norm(result.augmentdf[1, :residuals] / -5.614 - 1) < precision
	#@test result.r2_within > 0.0

	# subset
	model = @model Sales ~ Price ife = (pState + pYear, 2) fe = pState subset = (State .<= 30)
	result = regife(df, model, method = method, save = true)
	@test size(result.augmentdf, 1) == size(df, 1)
	@test norm(abs(result.augmentdf[1, :factors1]) / 0.25965 - 1) < precision
	@test norm(abs(result.augmentdf[1, :loadings1]) / 107.832 - 1) < precision
	@test norm(abs(result.augmentdf[1, :factors2])  / 0.2655  - 1) < precision
	@test norm(abs(result.augmentdf[1, :loadings2]) / 15.551 - 1) < precision
	@test norm(result.augmentdf[1, :residuals] / -3.8624  - 1) < precision
	@test norm(result.augmentdf[1, :pState] / 131.6162 - 1) < precision
end

# check high dimentional fixed effects are part of factor models
df.pState2 = categorical(df.State)
@test_throws ErrorException regife(df, @model(Sales ~ Price, ife = (pState + pYear, 2), weights = Pop, fe = pState2),  method = :levenberg_marquardt, save = true)


# local minima
using DataFrames, InteractiveFixedEffectModels, Test
precision = 2e-1


for method in [:levenberg_marquardt, :dogleg]
	println(method)

	df = CSV.read(joinpath(dirname(pathof(InteractiveFixedEffectModels)), "../dataset/Cigar.csv"))
	df.pState = categorical(df.State)
	df.pYear = categorical(df.Year)
	model = @model Sales ~ Price ife = (pState + pYear, 1) fe = pState weights = Pop
	result = regife(df, model, method = method, save = true)
	model = @model Sales ~ Price ife = (pState + pYear, 2) fe = pState weights = Pop
	result = regife(df, model, method = method, save = true)

	df = CSV.read(joinpath(dirname(pathof(InteractiveFixedEffectModels)), "../dataset/EmplUK.csv"))
	df.id1 = df.Firm
	df.id2 = df.Year
	df.pid1 = categorical(df.id1)
	df.pid2 = categorical(df.id2)
	df.y = df.Wage
	df.x1 = df.Emp
	df.w = df.Output
	model = @model y ~ x1 ife = (pid1 + pid2, 2) 
	result = regife(df, model, method = method, save = true)
	@test norm(result.coef ./ [4.53965, -0.0160858] .- 1) < precision
	model = @model y ~ x1 ife = (pid1 + pid2, 2) weights = w 
	result = regife(df, model, method = method, save = true)
	@test norm(result.coef ./ [3.47551,-0.017366] .- 1) < precision
	model = @model y ~ x1 ife = (pid1 + pid2, 1) weights = w 
	result = regife(df, model, method = method, save = true)
	@test norm(result.coef ./ [ -2.62105, -0.0470005] .- 1) < precision
end





