using RDatasets, DataFrames, SparseFactorModels, Base.Test
df = dataset("plm", "Cigar")
df[:pState] =  pool(df[:State])
df[:pYear] =  pool(df[:Year])

for method in [:svd, :ar, :cg]
	precision = 1e-1
	println(method)
	result = fit(SparseFactorModel(:pState, :pYear, 1), :Sales, df , method =  method, maxiter = 10_000) ;
	@test_approx_eq_eps abs(result.augmentdf[:factors1][1]) 0.18662770198472406 precision
	@test_approx_eq_eps abs(result.augmentdf[:loadings1][1]) 587.2272 precision
	@test_approx_eq_eps result.augmentdf[:residuals][1] -15.6928 precision

	result = fit(SparseFactorModel(:pState, :pYear, 2), :Sales,  df, method =  method, maxiter = 10_000) ;
	@test_approx_eq_eps abs(result.augmentdf[:factors1][1])  0.18662770198472406 precision
	@test_approx_eq_eps abs(result.augmentdf[:loadings1][1])  587.227 precision
	@test_approx_eq_eps result.augmentdf[:residuals][1] 2.16611 precision

	result = fit(SparseFactorModel(:pState, :pYear, 1), Sales ~ 1 |> pState, df, method = method, maxiter = 10_000) ;
	@test_approx_eq_eps abs(result.augmentdf[:factors1][1])   0.17636 precision
	@test_approx_eq_eps abs(result.augmentdf[:loadings1][1])  20.176432452716522 precision
	@test_approx_eq_eps result.augmentdf[:residuals][1]  -10.0181 precision
	@test_approx_eq_eps result.augmentdf[:pState][1]  107.4766 precision

	result = fit(SparseFactorModel(:pState, :pYear, 2), Sales ~ 1 |> pState, df, method =  method, maxiter = 10_000) ;
	@test_approx_eq_eps abs(result.augmentdf[:factors2][1]) 0.244 precision
	@test_approx_eq_eps abs(result.augmentdf[:loadings2][1]) 49.7943 precision
	@test_approx_eq_eps result.augmentdf[:residuals][1]  2.165319 precision
	@test_approx_eq_eps result.augmentdf[:pState][1]  107.47666 precision

	result = fit(SparseFactorModel(:pState, :pYear, 2), Sales ~ 1 |> pState, df, method =  method, subset = (df[:State] .<= 30), maxiter = 10_000) ;
	@test size(result.augmentdf, 1) == size(df, 1)
	@test_approx_eq_eps abs(result.augmentdf[:factors1][1])  0.20215 precision
	@test_approx_eq_eps abs(result.augmentdf[:loadings1][1]) 29.546 precision
	@test_approx_eq_eps abs(result.augmentdf[:factors2][1])  0.250768  precision
	@test_approx_eq_eps abs(result.augmentdf[:loadings2][1])   43.578 precision
	@test_approx_eq_eps result.augmentdf[:residuals][1]  2.9448  precision
	@test_approx_eq_eps result.augmentdf[:pState][1] 107.4766 precision

end
