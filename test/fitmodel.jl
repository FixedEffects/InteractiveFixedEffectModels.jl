using RDatasets, DataFrames, SparseFactorModels, Base.Test

df = dataset("plm", "Cigar")
df[:pState] = pool(df[:State])
df[:pYear] = pool(df[:Year])
method = :svd
precision = 1e-1


for method in [:svd, :gs, :gd]
	println(method)
	result = fit(SparseFactorModel(:pState, :pYear, 1), Sales ~ Price, df, method =  method)
	@test_approx_eq_eps result.coef [328.1653237715761, -1.0415042260420706]  precision
	@test_approx_eq_eps abs(result.augmentdf[1, :factors1]) 0.228 precision
	@test_approx_eq_eps abs(result.augmentdf[1, :loadings1]) 851.514 precision
	@test_approx_eq_eps result.augmentdf[1, :residuals] -9.7870 precision

	result = fit(SparseFactorModel(:pState, :pYear, 2), Sales ~ Price, df, method =  method)
	@test_approx_eq_eps result.coef [163.01350, -0.40610] 1e-2
	@test_approx_eq_eps abs(result.augmentdf[1, :factors1]) 0.227 precision
	@test_approx_eq_eps abs(result.augmentdf[1, :loadings1]) 184.1897 precision
	@test_approx_eq_eps result.augmentdf[1, :residuals]  -1.774 precision

	result = fit(SparseFactorModel(:pState, :pYear, 2), Sales ~ Price |> pState, df, method =  method)
	@test_approx_eq_eps result.coef -0.425389 1e-2
	@test_approx_eq_eps abs(result.augmentdf[1, :factors1])  0.2474 precision
	@test_approx_eq_eps abs(result.augmentdf[1, :loadings1]) 123.460 precision
	@test_approx_eq_eps result.augmentdf[1, :residuals] -5.222 precision

	result = fit(SparseFactorModel(:pState, :pYear, 2), Sales ~ Price |> pYear, df, method =  method)
	@test_approx_eq_eps result.coef -0.3744296120563005 1e-2
	@test_approx_eq_eps abs(result.augmentdf[1, :factors1])  0.1918 precision
	@test_approx_eq_eps abs(result.augmentdf[1, :loadings1]) 102.336 precision
	@test_approx_eq_eps result.augmentdf[1, :residuals] -2.2153 precision

	result = fit(SparseFactorModel(:pState, :pYear, 2), Sales ~ Price |> pState + pYear, df, method =  method)
	@test_approx_eq_eps result.coef -0.524157 precision
	@test_approx_eq_eps abs(result.augmentdf[1, :factors1]) 0.256 precision
	@test_approx_eq_eps abs(result.augmentdf[1, :loadings1]) 60.0481 precision
	@test_approx_eq_eps result.augmentdf[1, :residuals] -5.614 precision

	# subset
	result = fit(SparseFactorModel(:pState, :pYear, 2), Sales ~ Price |> pState, df, method =  method, subset = (df[:State] .<= 30)) 
	@test size(result.augmentdf, 1) == size(df, 1)
	@test_approx_eq_eps abs(result.augmentdf[:factors1][1]) 0.25965 precision
	@test_approx_eq_eps abs(result.augmentdf[:loadings1][1]) 107.832 precision
	@test_approx_eq_eps abs(result.augmentdf[:factors2][1])  0.2655  precision
	@test_approx_eq_eps abs(result.augmentdf[:loadings2][1])    15.551 precision
	@test_approx_eq_eps result.augmentdf[:residuals][1] -3.8624  precision
	@test_approx_eq_eps result.augmentdf[:pState][1] 131.6162 precision


	# test printing
	result = fit(SparseFactorModel(:pState, :pYear, 1), Sales ~ Price, df, method =  method)
	show(result)
end



#TODO: weight, subset, gradientdescent



