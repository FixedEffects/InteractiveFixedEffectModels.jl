```julia
using DataFrames, SparseFactorModels
N = 5000
T = 500
l1 = randn(N)
l2 = randn(N)
f1 = randn(T)
f2 = randn(T)
x1 = Array(Float64, N*T)
y = Array(Float64, N*T)
id = Array(Int64, N*T)
time = Array(Int64, N*T)

index = 0

function fillin(id, time, x1, y, N, T)
	index = 0
	@inbounds for i in 1:N
		for j in 1:T
			index += 1
			id[index] = i
			time[index] = j
			x1[index] = 4 + f1[j] * l1[i] + 3 * f2[j] * l2[i] + l1[i]^2 + randn()
			y[index] = 5 + 3 * x1[index] + 4 * f1[j] * l1[i] + f2[j] * l2[i] + randn()
		end
	end
end

fillin(id, time, x1, y, N, T)
df = DataFrame(id = pool(id), time = pool(time), x1 = x1, y = y)
subset = rand(1:5, N*T) .> 1
unbalanceddf = df[subset, :]
subset = rand(1:5, N*T) .== 1
sparsedf = df[subset, :]
```


# factor model 

### N x T
```julia
for method in [:gs, :gd, :momentum_gradient_descent, :gradient_descent, :cg, :svd]
	println("method : $(method)")
	@time result = fit(SparseFactorModel(:id, :time, 1), y ~ 1|> id + time, df, method = method, maxiter = 100_000)
	@show result.ess
	@show result.converged
end
```

Result
```
method : gs
elapsed time: 10.273916892 seconds (2864846148 bytes allocated, 21.97% gc time)
result.ess => 1.4309734432902223e8
result.converged => Bool[true]
method : gd
elapsed time: 24.828482005 seconds (2819128008 bytes allocated, 8.97% gc time)
result.ess => 1.4322765621668932e8
result.converged => Bool[true]
method : momentum_gradient_descent
elapsed time: 7.207087971 seconds (2763938764 bytes allocated, 31.20% gc time)
result.ess => 3.856913687839139e8
result.converged => Bool[true]
method : gdadient_descent
elapsed time: 7.23555626 seconds (2764864956 bytes allocated, 30.22% gc time)
result.ess => 3.8569136878281003e8
result.converged => Bool[true]
method : cg
elapsed time: 7.315325752 seconds (2770850608 bytes allocated, 30.48% gc time)
result.ess => 3.8569136878281003e8
result.converged => Bool[true]
method : svd
elapsed time: 7.408803942 seconds (2790164472 bytes allocated, 30.75% gc time)
result.ess => 1.4309734431613243e8
result.converged => Bool[true]
```

### N x T x 4/5
```julia
for method in [:gs, :gd, :momentum_gradient_descent, :gradient_descent, :cg, :svd]
	println("method : $(method)")
	@time result = fit(SparseFactorModel(:id, :time, 1), y ~ 1|> id + time, unbalanceddf, method = method, maxiter = 100_000, save = false)
	@show result.ess
	@show result.converged
end
```

Result
```
method : gs
elapsed time:  0.757044636 seconds (2265962964 bytes allocated, 27.72% gc time)
result.ess => 1.1448350015095846e8
result.converged => Bool[true]
method : gd
elapsed time: 22.210733998 seconds (2283018180 bytes allocated, 6.33% gc time)
result.ess => 1.1448362380288874e8
result.converged => Bool[true]
method : momentum_gradient_descent
elapsed time:  1.442086239 seconds (2293137652 bytes allocated, 24.74% gc time)
result.ess => 1.1448350012819447e8
result.converged => Bool[true]
method : gdadient_descent
elapsed time: 1.28273858 seconds (2282798676 bytes allocated, 26.14% gc time)
result.ess => 1.1448350013191424e8
result.converged => Bool[true]
method : cg
elapsed time:  0.395406888 seconds (2267540684 bytes allocated, 28.64% gc time)
result.ess => 3.0892192493692315e8
result.converged => Bool[true]
method : svd
elapsed time:  1.03806754  seconds (2311478892 bytes allocated, 26.66% gc time)
result.ess => 1.1448350012750378e8
result.converged => Bool[true]
```


### N x T x 1/5
```julia
for method in [:gs, :gd, :momentum_gradient_descent, :gradient_descent, :cg, :svd]
	println("method : $(method)")
	@time result = fit(SparseFactorModel(:id, :time, 1), y ~ 1|> id + time, sparsedf, method = method, maxiter = 100_000, save = false)
	@show result.ess
	@show result.converged
end
```

Result
```
method : gs
elapsed time: 0.227594335 seconds (35178344 bytes allocated, 25.16% gc time)
result.ess => 2.859276266826541e7
result.converged => Bool[true]
method : gd
elapsed time: 0.41333656 seconds (40195352 bytes allocated, 14.32% gc time)
result.ess => 2.8592762738720316e7
result.converged => Bool[true]
method : momentum_gradient_descent
elapsed time: 0.332329562 seconds (53835928 bytes allocated, 15.14% gc time)
result.ess => 2.8592762668812547e7
result.converged => Bool[true]
method : gdadient_descent
elapsed time: 0.374974405 seconds (50718128 bytes allocated, 13.42% gc time)
result.ess => 2.8592762678235393e7
result.converged => Bool[true]
method : cg
elapsed time: 0.121306865 seconds (36234960 bytes allocated, 42.80% gc time)
result.ess => 7.646351039796804e7
result.converged => Bool[true]
method : svd
elapsed time: 6.448935214 seconds (213958736 bytes allocated, 2.48% gc time)
result.ess => 2.8592762945056412e7
result.converged => Bool[true]
```



# interactive fixed effect 

### N x T

```julia
for method in [:gs, :gd, :svd]
	println("method : $(method)")
	@time result = fit(SparseFactorModel(:id, :time, 2), y ~ x1 |> id + time, df, method = method, maxiter = 10_000, save = false)
	@show result.ess
	@show result.converged
end
```

```
method : gs
elapsed time: 10.550448863 seconds (657691568 bytes allocated, 2.34% gc time)
result.ess => 2.482504231179842e6
result.converged => true
method : gd
elapsed time: 34.361296133 seconds (657729528 bytes allocated, 0.39% gc time)
result.ess => 2.482504252751984e6
result.converged => true
method : svd
elapsed time: 24.546804034 seconds (4781135024 bytes allocated, 15.07% gc time)
result.ess => 2.482504221441386e6
result.converged => true
```

### N x T x 4/5


```julia
for method in [:gs, :gd, :svd]
	println("method : $(method)")
	@time result = fit(SparseFactorModel(:id, :time, 2), y ~ x1 |> id + time, subdf, method = method, maxiter = 10_000, save = false)
	@show result.ess
	@show result.converged
end
```

```
method : gs
elapsed time: 11.220120599 seconds (527137368 bytes allocated, 2.19% gc time)
result.ess => 2.002572287830467e6
result.converged => true
method : gd
elapsed time: 28.092884058 seconds (527165760 bytes allocated)
result.ess => 2.0025721437885808e6
result.converged => true
method : svd
elapsed time: 28.171948893 seconds (4734809432 bytes allocated, 15.28% gc time)
result.ess => 2.001491168484043e6
result.converged => true
```

### N x T x 1/5


```julia
for method in [:gs, :gd, :svd]
	println("method : $(method)")
	@time result = fit(SparseFactorModel(:id, :time, 2), y ~ x1 |> id + time, sparsedf, method = method, maxiter = 10_000, save = false)
	@show result.ess
	@show result.converged
end
```

```
method : gs
elapsed time: 0.694661678 seconds (134055392 bytes allocated, 15.28% gc time)
result.ess => 545282.1325988211
result.converged => true
method : gd
elapsed time: 2.030669054 seconds (134065560 bytes allocated, 5.51% gc time)
result.ess => 545282.1330403608
result.converged => true
method : svd
elapsed time: 14.12032201 seconds (1104335920 bytes allocated, 7.59% gc time)
result.ess => 543869.2977804352
result.converged => true
```
