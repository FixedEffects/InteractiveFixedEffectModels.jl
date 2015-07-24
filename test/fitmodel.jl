using RDatasets, DataFrames, PanelFactorModels, Base.Test

df = dataset("plm", "Cigar")
df[:pState] = pool(df[:State])
df[:pYear] = pool(df[:Year])
method = :svd

for method in [:svd, :gs, :bfgs]
	println(method)
	result = fit(PanelFactorModel(:pState, :pYear, 1), Sales ~ Price, df, method =  method) ;
	@test_approx_eq_eps result.coef [328.1653237715761, -1.0415042260420706]  1e-1
	result = fit(PanelFactorModel(:pState, :pYear, 2), Sales ~ Price, df, method =  method) ;
	@test_approx_eq_eps result.coef [163.01350, -0.40610] 1e-2
	result = fit(PanelFactorModel(:pState, :pYear, 2), Sales ~ Price |> pState, df, method =  method) ;
	@test_approx_eq_eps result.coef -0.425389 1e-2
	result = fit(PanelFactorModel(:pState, :pYear, 2), Sales ~ Price |> pYear, df, method =  method) ;
	@test_approx_eq_eps result.coef -0.3744296120563005 1e-2
	result = fit(PanelFactorModel(:pState, :pYear, 2), Sales ~ Price |> pState + pYear, df, method =  method) ;
	@test_approx_eq_eps result.coef -0.524157 1e-2
	
	# check normalization F'F = Id 
	@test_approx_eq result.factors' * result.factors  eye(size(result.factors, 2))
	cross = result.loadings' * result.loadings
	@test_approx_eq cross diagm(diag(cross))
end







