using RDatasets, DataFrames, PanelFactorModels, Base.Test

df = dataset("plm", "Cigar")
df[:pState] = pool(df[:State])
df[:pYear] = pool(df[:Year])


# test coef
for method in [:bfgs, :svd]
	result = fit(PanelFactorModel(:pState, :pYear, 2), Sales ~ Price, df, tol = 1e-9, maxiter = 10000, method = method)
	@test_approx_eq_eps result.coef [163.01350,-0.40610] 1e-2

	result = fit(PanelFactorModel(:pState, :pYear, 2), Sales ~ Price + Pop, df, tol = 1e-9, maxiter = 10000, method = method)

	@test_approx_eq_eps result.coef [161.47626,-0.41099,0.00042] 1e-2


	# test coef with absorb option
	@test_approx_eq_eps fit(PanelFactorModel(:pState, :pYear, 2),  Sales ~ Price |> pState, df , tol = 1e-9, method = method).coef  [-0.42538] 1e-2


	# check normalization F'F = Id 
	@test_approx_eq result.factors' * result.factors  eye(size(result.factors, 2))
	# Check Lambda' Lambda = diag

	cross = result.loadings' * result.loadings
	@test_approx_eq cross diagm(diag(cross))
end





