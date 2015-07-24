using RDatasets, DataFrames, PanelFactorModels, Base.Test
df = dataset("plm", "Cigar")
df[:pState] =  pool(df[:State])
df[:pYear] =  pool(df[:Year])
method = :svd


for method in [:svd, :backpropagation, :gradient_descent]
	println(method)
	result = fit(PanelFactorModel(:pState, :pYear, 1), :Sales, df, method =  method) ;
	@test_approx_eq_eps abs(result.factors[1]) 0.18662770198472406 1e-2

	if method != :backpropagation
		result = fit(PanelFactorModel(:pState, :pYear, 2), :Sales, df, method =  method) ;
		@test_approx_eq_eps abs(result.factors[1])  0.18662770198472406 1e-1
	end
	
	result = fit(PanelFactorModel(:pState, :pYear, 1), Sales ~ 1 |> pState, df, method =  method) ;
	@test_approx_eq_eps abs(result.factors[1])   0.17636 1e-1

	result = fit(PanelFactorModel(:pState, :pYear, 1), Sales ~ 1 |> pYear, df, method =  method) ;
	@test_approx_eq_eps abs(result.factors[1])   0.185338  1e-1


	# check normalization F'F = Id 
	@test_approx_eq result.factors' * result.factors  eye(size(result.factors, 2))
	cross = result.loadings' * result.loadings
	@test_approx_eq cross diagm(diag(cross))
end
