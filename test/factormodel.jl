using DataFrames, InteractiveFixedEffectModels, Base.Test
df = readtable(joinpath(dirname(@__FILE__), "..", "dataset", "Cigar.csv.gz"))
df[:pState] =  pool(df[:State])
df[:pYear] =  pool(df[:Year])
method = :gauss_seidel

for method in [:gauss_seidel,  :dogleg, :levenberg_marquardt]
	precision = 1e-1
	println(method)
	result = reg(df, @formula(Sales ~ 0), @ife(pState + pYear, 1), method =  method, maxiter = 10_000) ;
	@test abs(result.augmentdf[:factors1][1]) ≈ 0.18662770198472406 atol = precision
	@test abs(result.augmentdf[:loadings1][1]) ≈ 587.2272 atol = precision
	@test result.augmentdf[:residuals][1] ≈ - 15.6928 atol = precision

	result = reg(df, @formula(Sales ~ 0), @ife(pState + pYear, 2), method =  method, maxiter = 10_000) ;
	@test abs(result.augmentdf[:factors1][1])  ≈ 0.18662770198472406 atol = precision
	@test abs(result.augmentdf[:loadings1][1])  ≈ 587.227 atol = precision
	@test result.augmentdf[:residuals][1] ≈ 2.16611 atol = precision

	result = reg(df, @formula(Sales ~ 0), @ife(pState + pYear, 1), @fe(pState), method = method, maxiter = 10_000) ;
	@test abs(result.augmentdf[:factors1][1])   ≈ 0.17636 atol = precision
	@test abs(result.augmentdf[:loadings1][1])  ≈ 20.176432452716522 atol = precision
	@test result.augmentdf[:residuals][1]  ≈ - 10.0181 atol = precision
	@test result.augmentdf[:pState][1]  ≈ 107.4766 atol = precision

	result = reg(df, @formula(Sales ~ 0), @ife(pState + pYear, 2), @fe(pState), method =  method, maxiter = 10_000) ;
	@test abs(result.augmentdf[:factors2][1]) ≈ 0.244 atol = precision
	@test abs(result.augmentdf[:loadings2][1]) ≈ 49.7943 atol = precision
	@test result.augmentdf[:residuals][1]  ≈ 2.165319 atol = precision
	@test result.augmentdf[:pState][1]  ≈ 107.47666 atol = precision

	result = reg(df, @formula(Sales ~ 0), @ife(pState + pYear, 2), @fe(pState), method =  method, subset = (df[:State] .<= 30), maxiter = 10_000) ;
	@test size(result.augmentdf, 1) == size(df, 1)
	@test abs(result.augmentdf[:factors1][1])  ≈ 0.20215 atol = precision
	@test abs(result.augmentdf[:loadings1][1]) ≈ 29.546 atol = precision
	@test abs(result.augmentdf[:factors2][1])  ≈ 0.250768  atol = precision
	@test abs(result.augmentdf[:loadings2][1])   ≈ 43.578 atol = precision
	@test result.augmentdf[:residuals][1]  ≈ 2.9448  atol = precision
	@test result.augmentdf[:pState][1] ≈ 107.4766 atol = precision

end
