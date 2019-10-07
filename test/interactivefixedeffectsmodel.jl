using DataFrames, InteractiveFixedEffectModels, CSV, LinearAlgebra, Test

precision = 2e-1
df = CSV.read(joinpath(dirname(pathof(InteractiveFixedEffectModels)), "../dataset/Cigar.csv"))



for method in [:dogleg, :levenberg_marquardt, :gauss_seidel]
	println(method)
	model = @formula Sales ~ Price + ife(State, Year, 1)
	result = regife(df, model, method = method, save = true)
	@test norm(result.coef ./ [328.1653237715761, -1.0415042260420706] .- 1)  < precision
	@test norm(abs(result.augmentdf[1, :factors1]) / 0.228 - 1) < precision
	@test norm(abs(result.augmentdf[1, :loadings1]) / 851.514 - 1) < precision
	@test norm(result.augmentdf[1, :residuals] / -9.7870 - 1) < precision
	@test result.r2_within > 0.0

	show(result)
	model = @formula Sales ~ Price + ife(State, Year, 2)
	result = regife(df, model, method = method, save = true)
	@test norm(result.coef ./ [163.01350, -0.40610] .- 1) < precision
	@test norm(abs(result.augmentdf[1, :factors1]) / 0.227 - 1) < precision
	@test norm(abs(result.augmentdf[1, :loadings1]) / 184.1897 - 1) < precision
	@test norm(result.augmentdf[1, :residuals]  / -1.774 - 1) < precision
	@test result.r2_within > 0.0

	model = @formula Sales ~ Price + ife(State, Year, 2) + fe(State)
	result = regife(df, model, method = method, save = true)
	@test norm(result.coef ./ -0.425389 .- 1) < precision
	@test norm(abs(result.augmentdf[1, :factors1])  / 0.2474 - 1) < precision
	@test norm(abs(result.augmentdf[1, :loadings1]) / 123.460 - 1) < precision
	@test norm(result.augmentdf[1, :residuals] / -5.222 - 1) < precision
	@test result.r2_within > 0.0

	model = @formula Sales ~ Price + ife(State, Year, 2) + fe(Year)
	result = regife(df, model, method = method, save = true)
	@test norm(result.coef ./ -0.3744296120563005 .- 1) < precision
	@test norm(abs(result.augmentdf[1, :factors1])  / 0.1918 - 1) < precision
	@test norm(abs(result.augmentdf[1, :loadings1]) / 102.336 - 1) < precision
	@test norm(result.augmentdf[1, :residuals] / -2.2153 - 1) < precision
	@test result.r2_within > 0.0

	model = @formula Sales ~ Price + ife(State, Year, 2) + fe(State) + fe(Year)
	result = regife(df, model, method = method, save = true)
	@test norm(result.coef ./ -0.524157 .- 1) < precision
	@test norm(abs(result.augmentdf[1, :factors1]) / 0.256 - 1) < precision
	@test norm(abs(result.augmentdf[1, :loadings1]) / 60.0481 - 1) < precision
	@test norm(result.augmentdf[1, :residuals] / -5.614 - 1) < precision
	@test result.r2_within > 0.0

	# subset
	model = @formula Sales ~ Price + ife(State, Year, 2) + fe(State) 
	result = regife(df, model, method = method, subset = df.State .<= 30, save = true)
	@test size(result.augmentdf, 1) == size(df, 1)
	@test norm(abs(result.augmentdf[1, :factors1]) / 0.25965 - 1) < precision
	@test norm(abs(result.augmentdf[1, :loadings1]) / 107.832 - 1) < precision
	@test norm(abs(result.augmentdf[1, :factors2])  / 0.2655  - 1) < precision
	@test norm(abs(result.augmentdf[1, :loadings2]) / 15.551 - 1) < precision
	@test norm(result.augmentdf[1, :residuals] / -3.8624  - 1) < precision
	@test norm(result.augmentdf[1, :fe_State] / 131.6162 - 1) < precision
end

# check high dimentional fixed effects are part of factor models
df.State2 = categorical(df.State)
@test_throws ErrorException regife(df, @formula(Sales ~ Price + ife(State, Year, 2) + fe(State2)))


# local minima
using DataFrames, InteractiveFixedEffectModels, Test
precision = 2e-1


for method in [:levenberg_marquardt, :dogleg]
	println(method)

	df = CSV.read(joinpath(dirname(pathof(InteractiveFixedEffectModels)), "../dataset/Cigar.csv"))
	df.State = categorical(df.State)
	df.Year = categorical(df.Year)
	model = @formula Sales ~ Price + ife(State, Year, 1) + fe(State)
	result = regife(df, model, weights = :Pop, method = method, save = true)
	model = @formula Sales ~ Price + ife(State, Year, 2) + fe(State)
	result = regife(df, model, weights = :Pop, method = method, save = true)

	df = CSV.read(joinpath(dirname(pathof(InteractiveFixedEffectModels)), "../dataset/EmplUK.csv"))
	df.id1 = df.Firm
	df.id2 = df.Year
	df.pid1 = categorical(df.id1)
	df.pid2 = categorical(df.id2)
	df.y = df.Wage
	df.x1 = df.Emp
	df.w = df.Output
	model = @formula y ~ x1 + ife(pid1, pid2, 2) 
	result = regife(df, model, method = method, save = true)
	@test norm(result.coef ./ [4.53965, -0.0160858] .- 1) < precision
	model = @formula y ~ x1 + ife(pid1, pid2, 2)
	result = regife(df, model, weights = :w, method = method, save = true)
	@test norm(result.coef ./ [3.47551,-0.017366] .- 1) < precision
	model = @formula y ~ x1 + ife(pid1, pid2, 1)
	result = regife(df, model, weights = :w, method = method, save = true)
	@test norm(result.coef ./ [ -2.62105, -0.0470005] .- 1) < precision
end

# Check old syntax still works
df[!, :pState] = categorical(df.State)
df[!, :pYear] = categorical(df.Year)
model = @model Sales ~ Price ife = (pState + pYear, 2) fe = pState
result = regife(df, model)

# Check formula can be constructed programatically
using StatsModels
result1 = regife(df, @formula(Sales ~ NDI + fe(State) + ife(State, Year, 2)))
result2 = regife(df, Term(:Sales) ~ Term(:NDI) + fe(Term(:State)) + ife(Term(:State), Term(:Year), 2))
@test result1.coef ≈ result2.coef


