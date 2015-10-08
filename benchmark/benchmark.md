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
for method in [:levenberg_marquardt, :dogleg, :gauss_seidel, :regar]
	println("method : $(method)")
	@time result = fit(SparseFactorModel(:id, :time, 2), y ~ x1 |> id + time, df, method = method, save = false)
	@show result.iterations
	@show result.ess
	@show result.converged
end
```

```
method : levenberg_marquardt
 19.741259 seconds (58.97 M allocations: 1.621 GB, 3.10% gc time)
result.iterations = 371
result.ess = 2.4806926953510325e6
result.converged = true
method : dogleg
INFO: Algorithm ended up on a local minimum. Restarting from a new, random, x0.
 12.594733 seconds (12.63 M allocations: 1.174 GB, 5.74% gc time)
result.iterations = 212
result.ess = 2.4806926953499895e6
result.converged = true
method : gauss_seidel
 8.166197 seconds (706.90 k allocations: 750.712 MB, 4.39% gc time)
result.iterations = 57
result.ess = 2.4806927020173306e6
result.converged = true
method : regar
  8.466838 seconds (7.89 M allocations: 976.448 MB, 6.37% gc time)
result.iterations = 187
result.ess = 2.4806926953436537e6
result.converged = true
```

### N x T x 4/5


```julia
for method in [:levenberg_marquardt, :dogleg, :gauss_seidel, :regar]
	println("method : $(method)")
	@time result = fit(SparseFactorModel(:id, :time, 2), y ~ x1 |> id + time, unbalanceddf, method = method, save = false)
	@show result.iterations
	@show result.ess
	@show result.converged
end
```

```
method : levenberg_marquardt
 15.968084 seconds (55.60 M allocations: 1.423 GB, 3.91% gc time)
result.iterations = 365
result.ess = 1.997676617351034e6
result.converged = true
method : dogleg
 33.113829 seconds (59.65 M allocations: 1.483 GB, 1.68% gc time)
result.iterations = 1104
result.ess = 1.9976766273685207e6
result.converged = true
method : gauss_seidel
  9.755387 seconds (708.27 k allocations: 603.140 MB, 3.70% gc time)
result.iterations = 81
result.ess = 1.9986716369475075e6
result.converged = true
method : regar
 29.806383 seconds (54.87 M allocations: 1.352 GB, 1.55% gc time)
result.iterations = 1515
result.ess = 1.9976766236881565e6
result.converged = true
```

### N x T x 1/5


```julia
for method in [:levenberg_marquardt, :dogleg, :gauss_seidel, :regar]
	println("method : $(method)")
	@time result = fit(SparseFactorModel(:id, :time, 2), y ~ x1 |> id + time, sparsedf, method = method, save = false)
	@show result.iterations
	@show result.ess
	@show result.converged
end
```

```
method : levenberg_marquardt
  5.169143 seconds (55.01 M allocations: 993.576 MB, 4.67% gc time)
result.iterations = 358
result.ess = 542208.191393264
result.converged = true
method : dogleg
  9.782926 seconds (66.56 M allocations: 1.143 GB, 2.40% gc time)
result.iterations = 1226
result.ess = 542208.1914324684
result.converged = true
method : gauss_seidel
  1.180293 seconds (704.51 k allocations: 160.071 MB, 7.41% gc time)
result.iterations = 15
result.ess = 543565.5889823908
result.converged = true
method : regar
 17.250520 seconds (114.28 M allocations: 1.838 GB, 1.57% gc time)
result.iterations = 3589
result.ess = 542208.191398862
result.converged = true
```
