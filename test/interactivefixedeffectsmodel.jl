using RDatasets, DataFrames, InteractiveFixedEffectModels, Distances, Base.Test

precision = 2e-1
df = dataset("plm", "Cigar")
df[:pState] = pool(df[:State])
df[:pYear] = pool(df[:Year])
method = :levenberg_marquardt

for method in [:levenberg_marquardt, :dogleg, :gauss_seidel]
	println(method)


	result = reg(df, @formula(Sales ~ Price), @ife(pState + pYear, 1), method = method, save = true)
	@test norm(result.coef ./ [328.1653237715761, -1.0415042260420706] .- 1)  < precision
	@test norm(abs(result.augmentdf[1, :factors1]) / 0.228 - 1) < precision
	@test norm(abs(result.augmentdf[1, :loadings1]) / 851.514 - 1) < precision
	@test norm(result.augmentdf[1, :residuals] / -9.7870 - 1) < precision
	@test result.r2_within > 0.0

	show(result)
	result = reg(df, @formula(Sales ~ Price), @ife(pState + pYear, 2), method = method, save = true)
	@test norm(result.coef ./ [163.01350, -0.40610] - 1) < precision
	@test norm(abs(result.augmentdf[1, :factors1]) / 0.227 - 1) < precision
	@test norm(abs(result.augmentdf[1, :loadings1]) / 184.1897 - 1) < precision
	@test norm(result.augmentdf[1, :residuals]  / -1.774 - 1) < precision
	@test result.r2_within > 0.0

	result = reg(df, @formula(Sales ~ Price), @ife(pState + pYear, 2), @fe(pState), method = method, save = true)
	@test norm(result.coef / -0.425389 - 1) < precision
	@test norm(abs(result.augmentdf[1, :factors1])  / 0.2474 - 1) < precision
	@test norm(abs(result.augmentdf[1, :loadings1]) / 123.460 - 1) < precision
	@test norm(result.augmentdf[1, :residuals] / -5.222 - 1) < precision
	@test result.r2_within > 0.0

	result = reg(df, @formula(Sales ~ Price), @ife(pState + pYear, 2), @fe(pYear), method = method, save = true)
	@test norm(result.coef / -0.3744296120563005 -1 ) < precision
	@test norm(abs(result.augmentdf[1, :factors1])  / 0.1918 - 1) < precision
	@test norm(abs(result.augmentdf[1, :loadings1]) / 102.336 - 1) < precision
	@test norm(result.augmentdf[1, :residuals] / -2.2153 - 1) < precision
	@test result.r2_within > 0.0

	result = reg(df, @formula(Sales ~ Price), @ife(pState + pYear, 2), @fe(pState + pYear), method = method, save = true)
	@test norm(result.coef / -0.524157 - 1) < precision
	@test norm(abs(result.augmentdf[1, :factors1]) / 0.256 - 1) < precision
	@test norm(abs(result.augmentdf[1, :loadings1]) / 60.0481 - 1) < precision
	@test norm(result.augmentdf[1, :residuals] / -5.614 - 1) < precision
	@test result.r2_within > 0.0

	# subset
	result = reg(df, @formula(Sales ~ Price), @ife(pState + pYear, 2), @fe(pState), method = method, subset = (df[:State] .<= 30), save = true)
	@test size(result.augmentdf, 1) == size(df, 1)
	@test norm(abs(result.augmentdf[:factors1][1]) /0.25965 - 1) < precision
	@test norm(abs(result.augmentdf[:loadings1][1]) /107.832 - 1) < precision
	@test norm(abs(result.augmentdf[:factors2][1])  /0.2655  - 1) < precision
	@test norm(abs(result.augmentdf[:loadings2][1])    /15.551 - 1) < precision
	@test norm(result.augmentdf[:residuals][1] /-3.8624  - 1) < precision
	@test norm(result.augmentdf[:pState][1] /131.6162 - 1) < precision
end

# check high dimentional fixed effects are part of factor models
df[:pState2] = pool(df[:State])
@test_throws ErrorException reg(df, @formula(Sales ~ Price), @ife(pState + pYear, 2), @fe(pState2), @weight(Pop), method = method, save = true)


# local minima
using RDatasets, DataFrames, InteractiveFixedEffectModels, Distances, Base.Test
precision = 2e-1

for method in [:levenberg_marquardt, :dogleg]
	println(method)

	df = dataset("plm", "Cigar")
	df[:pState] = pool(df[:State])
	df[:pYear] = pool(df[:Year])
	result = reg(df, @formula(Sales ~ Price), @ife(pState + pYear, 1), @fe(pState), @weight(Pop), method = method, save = true)
	result = reg(df, @formula(Sales ~ Price), @ife(pState + pYear, 2), @fe(pState), @weight(Pop), method = method, save = true)

	df = dataset("plm", "EmplUK")
	df[:id1] = df[:Firm]
	df[:id2] = df[:Year]
	df[:pid1] = pool(df[:id1])
	df[:pid2] = pool(df[:id2])
	df[:y] = df[:Wage]
	df[:x1] = df[:Emp]
	df[:w] = df[:Output]
	result = reg(df, @formula(y ~ x1), @ife(pid1 + pid2, 2), method = method, save = true)
	@test norm(result.coef ./ [4.53965, -0.0160858] - 1) < precision
	result = reg(df, @formula(y ~ x1), @ife(pid1 + pid2, 2), @weight(w), method = method, save = true)
	@test norm(result.coef ./ [3.47551,-0.017366] - 1) < precision
	result = reg(df, @formula(y ~ x1), @ife(pid1 + pid2, 1), @weight(w), method = method, save = true)
	@test norm(result.coef ./ [ -2.62105, -0.0470005] - 1) < precision
end





