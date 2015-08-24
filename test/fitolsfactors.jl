using RDatasets, DataFrames, SparseFactorModels, Distances, Base.Test

df = dataset("plm", "Cigar")
df[:pState] = pool(df[:State])
df[:pYear] = pool(df[:Year])
method = :svd
precision = 2e-1



#TODO: weight, subset, gradientdescent

for method in [:svd, :ar, :gd]
	println(method)
	result = fit(SparseFactorModel(:pState, :pYear, 1), Sales ~ Price, df, method =  method, maxiter = 10_000)
	@test norm(result.coef ./ [328.1653237715761, -1.0415042260420706] .- 1)  < precision
	@test norm(abs(result.augmentdf[1, :factors1]) / 0.228 - 1) < precision
	@test norm(abs(result.augmentdf[1, :loadings1]) / 851.514 - 1) < precision
	@test norm(result.augmentdf[1, :residuals] / -9.7870 - 1) < precision

	result = fit(SparseFactorModel(:pState, :pYear, 2), Sales ~ Price, df, method =  method, maxiter = 10_000)
	@test norm(result.coef ./ [163.01350, -0.40610] - 1) < precision
	@test norm(abs(result.augmentdf[1, :factors1]) / 0.227 - 1) < precision
	@test norm(abs(result.augmentdf[1, :loadings1]) / 184.1897 - 1) < precision
	@test norm(result.augmentdf[1, :residuals]  / -1.774 - 1) < precision

	result = fit(SparseFactorModel(:pState, :pYear, 2), Sales ~ Price |> pState, df, method =  method, maxiter = 1000)
	@test norm(result.coef / -0.425389 - 1) < precision
	@test norm(abs(result.augmentdf[1, :factors1])  / 0.2474 - 1) < precision
	@test norm(abs(result.augmentdf[1, :loadings1]) / 123.460 - 1) < precision
	@test norm(result.augmentdf[1, :residuals] / -5.222 - 1) < precision

	result = fit(SparseFactorModel(:pState, :pYear, 2), Sales ~ Price |> pYear, df, method =  method, maxiter = 1000)
	@test norm(result.coef / -0.3744296120563005 -1 ) < precision
	@test norm(abs(result.augmentdf[1, :factors1])  / 0.1918 - 1) < precision
	@test norm(abs(result.augmentdf[1, :loadings1]) / 102.336 - 1) < precision
	@test norm(result.augmentdf[1, :residuals] / -2.2153 - 1) < precision

	result = fit(SparseFactorModel(:pState, :pYear, 2), Sales ~ Price |> pState + pYear, df, method =  method, maxiter = 1000)
	@test norm(result.coef / -0.524157 - 1) < precision
	@test norm(abs(result.augmentdf[1, :factors1]) / 0.256 - 1) < precision
	@test norm(abs(result.augmentdf[1, :loadings1]) / 60.0481 - 1) < precision
	@test norm(result.augmentdf[1, :residuals] / -5.614 - 1) < precision

	# subset
	result = fit(SparseFactorModel(:pState, :pYear, 2), Sales ~ Price |> pState, df, method =  method, subset = (df[:State] .<= 30), maxiter = 1000)
	@test size(result.augmentdf, 1) == size(df, 1)
	@test norm(abs(result.augmentdf[:factors1][1]) /0.25965 - 1) < precision
	@test norm(abs(result.augmentdf[:loadings1][1]) /107.832 - 1) < precision
	@test norm(abs(result.augmentdf[:factors2][1])  /0.2655  - 1) < precision
	@test norm(abs(result.augmentdf[:loadings2][1])    /15.551 - 1) < precision
	@test norm(result.augmentdf[:residuals][1] /-3.8624  - 1) < precision
	@test norm(result.augmentdf[:pState][1] /131.6162 - 1) < precision

	# show method
	result = fit(SparseFactorModel(:pState, :pYear, 1), Sales ~ Price, df, method =  method, maxiter = 1000)
	show(result)
end
