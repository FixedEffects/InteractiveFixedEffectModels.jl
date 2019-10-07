using DataFrames, InteractiveFixedEffectModels,Test, CSV
df = df = CSV.read(joinpath(dirname(pathof(InteractiveFixedEffectModels)), "../dataset/Cigar.csv"))

for method in [:gauss_seidel,  :dogleg, :levenberg_marquardt]
	println(method)
	model = @formula Sales ~ 0 + ife(State, Year, 1)
	result = regife(df, model, method = method, maxiter = 10_000)
	@test abs(result.augmentdf[1, :factors1]) ≈ 0.18662770198472406 atol = 1e-1
	@test abs(result.augmentdf[1, :loadings1]) ≈ 587.2272 atol = 1e-1
	@test result.augmentdf[1, :residuals] ≈ - 15.6928 atol = 1e-1

	model = @formula Sales ~ 0 + ife(State, Year, 2)
	result = regife(df, model, method = method, maxiter = 10_000)
	@test abs(result.augmentdf[1, :factors1])  ≈ 0.18662770198472406 atol = 1e-1
	@test abs(result.augmentdf[1, :loadings1])  ≈ 587.227 atol = 1e-1
	@test result.augmentdf[1, :residuals] ≈ 2.16611 atol = 1e-1

	model = @formula Sales ~ ife(State, Year, 1) + fe(State)
	result = regife(df, model, method = method, maxiter = 10_000)
	@test abs(result.augmentdf[1, :factors1])   ≈ 0.17636 atol = 1e-1
	@test abs(result.augmentdf[1, :loadings1])  ≈ 20.176432452716522 atol = 1e-1
	@test result.augmentdf[1, :residuals]  ≈ - 10.0181 atol = 1e-1
	@test result.augmentdf[1, :fe_State]  ≈ 107.4766 atol = 1e-1

	model = @formula Sales ~ ife(State, Year, 2) + fe(State)
	result = regife(df, model, method = method, maxiter = 10_000)
	@test abs(result.augmentdf[1, :factors2]) ≈ 0.244 atol = 1e-1
	@test abs(result.augmentdf[1, :loadings2]) ≈ 49.7943 atol = 1e-1
	@test result.augmentdf[1, :residuals]  ≈ 2.165319 atol = 1e-1
	@test result.augmentdf[1, :fe_State]  ≈ 107.47666 atol = 1e-1

	model = @formula Sales ~ ife(State, Year, 2) + fe(State)
	result = regife(df, model, subset = df.State .<= 30, method = method, maxiter = 10_000)
	@test size(result.augmentdf, 1) == size(df, 1)
	@test abs(result.augmentdf[1, :factors1])  ≈ 0.20215 atol = 1e-1
	@test abs(result.augmentdf[1, :loadings1]) ≈ 29.546 atol = 1e-1
	@test abs(result.augmentdf[1, :factors2])  ≈ 0.250768  atol = 1e-1
	@test abs(result.augmentdf[1, :loadings2])   ≈ 43.578 atol = 1e-1
	@test result.augmentdf[1, :residuals]  ≈ 2.9448  atol = 1e-1
	@test result.augmentdf[1, :fe_State] ≈ 107.4766 atol = 1e-1
end
