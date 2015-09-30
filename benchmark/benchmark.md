```julia
using DataFrames, SparseFactorModels
srand(1234)
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
for method in [:levenberg_marquardt, :dogleg, :gauss_seidel]
	println("method : $(method)")
	@time result = fit(SparseFactorModel(:id, :time, 1), y ~ 1|> id + time, df, method = method, save = false)
	@show result.iterations
	@show result.ess
	@show result.converged
end
```

Result
```
method : levenberg_marquardt
  7.668749 seconds (19.03 M allocations: 513.302 MB, 2.82% gc time)
result.iterations = 188
result.ess = 1.404830098859686e8
result.converged = true
method : dogleg
  1.048312 seconds (477.81 k allocations: 230.311 MB, 11.47% gc time)
result.iterations = 22
result.ess = 3.7333837177100134e8
result.converged = true
method : gauss_seidel
  0.831421 seconds (984 allocations: 146.277 MB, 3.18% gc time)
result.iterations = 18
result.ess = 1.4048300982167023e8
result.converged = true
```

### N x T x 4/5
```julia
for method in [:levenberg_marquardt, :dogleg, :gauss_seidel]
	println("method : $(method)")
	@time result = fit(SparseFactorModel(:id, :time, 1), y ~ 1|> id + time, unbalanceddf, method = method, save = false)
	@show result.iterations
	@show result.ess
	@show result.converged
end
```

Result
```
method : levenberg_marquardt
  5.529451 seconds (17.50 M allocations: 445.647 MB, 3.76% gc time)
result.iterations = 194
result.ess = 1.123447290001289e8
result.converged = true
method : dogleg
  6.351485 seconds (7.42 M allocations: 291.833 MB, 2.84% gc time)
result.iterations = 304
result.ess = 1.1234472899117553e8
result.converged = true
method : gauss_seidel
  0.665716 seconds (961 allocations: 117.137 MB, 0.80% gc time)
result.iterations = 15
result.ess = 1.123447289365168e8
result.converged = true
```


### N x T x 1/5
```julia
for method in [:levenberg_marquardt, :dogleg, :gauss_seidel]
	println("method : $(method)")
	@time result = fit(SparseFactorModel(:id, :time, 1), y ~ 1|> id + time, sparsedf, method = method, save = false)
	@show result.iterations
	@show result.ess
	@show result.converged
end
```

Result
```
method : levenberg_marquardt
  2.411380 seconds (26.65 M allocations: 451.970 MB, 2.91% gc time)
result.iterations = 322
result.ess = 2.810035008926143e7
result.converged = true
method : dogleg
  1.555647 seconds (5.78 M allocations: 133.536 MB, 6.08% gc time)
result.iterations = 272
result.ess = 2.810035008495609e7
result.converged = true
method : gauss_seidel
  0.210676 seconds (994 allocations: 29.688 MB, 0.66% gc time)
result.iterations = 19
result.ess = 2.810035006800005e7
result.converged = true
```



# interactive fixed effect 

### N x T

```julia
for method in [:levenberg_marquardt, :dogleg, :gauss_seidel]
	println("method : $(method)")
	@time result = fit(SparseFactorModel(:id, :time, 2), y ~ x1 |> id + time, df, method = method, save = false)
	@show result.iterations
	@show result.ess
	@show result.converged
end
```

```
method : levenberg_marquardt
 12.515678 seconds (35.65 M allocations: 1.273 GB, 3.60% gc time)
result.iterations = 224
result.ess = 2.4806926953467345e6
result.converged = true
method : dogleg
 10.653673 seconds (11.77 M allocations: 939.787 MB, 4.96% gc time)
result.iterations = 222
result.ess = 2.4806926954198824e6
result.converged = true
method : gauss_seidel
  8.087318 seconds (3.91 k allocations: 644.233 MB, 4.22% gc time)
result.iterations = 62
result.ess = 2.48069270035986e6
result.converged = true
```

### N x T x 4/5


```julia
for method in [:levenberg_marquardt, :dogleg, :gauss_seidel]
	println("method : $(method)")
	@time result = fit(SparseFactorModel(:id, :time, 2), y ~ x1 |> id + time, unbalanceddf, method = method, maxiter = 10_000, save = false)
	@show result.iterations
	@show result.ess
	@show result.converged
end
```

```
method : levenberg_marquardt
 20.928472 seconds (74.36 M allocations: 1.702 GB, 3.32% gc time)
result.iterations = 472
result.ess = 1.9976766172168502e6
result.converged = true
method : dogleg
 39.458185 seconds (66.72 M allocations: 1.589 GB, 1.58% gc time)
result.iterations = 1256
result.ess = 1.9976766172250109e6
result.converged = true
method : gauss_seidel
  9.025863 seconds (4.64 k allocations: 515.717 MB, 2.99% gc time)
result.iterations = 82
result.ess = 1.9986716371922768e6
result.converged = true
```

### N x T x 1/5


```julia
for method in [:levenberg_marquardt, :dogleg, :gauss_seidel]
	println("method : $(method)")
	@time result = fit(SparseFactorModel(:id, :time, 2), y ~ x1 |> id + time, sparsedf, method = method, maxiter = 10_000, save = false)
	@show result.iterations
	@show result.ess
	@show result.converged
end
```

```
method : levenberg_marquardt
  5.687987 seconds (62.94 M allocations: 1.088 GB, 5.60% gc time)
result.iterations = 414
result.ess = 542208.1913574268
result.converged = true
method : dogleg
  7.166447 seconds (46.32 M allocations: 861.144 MB, 2.89% gc time)
result.iterations = 862
result.ess = 542208.1913257103
result.converged = true
method : gauss_seidel
  1.212241 seconds (2.51 k allocations: 129.907 MB, 2.48% gc time)
result.iterations = 21
result.ess = 543565.5891925972
result.converged = true
```
