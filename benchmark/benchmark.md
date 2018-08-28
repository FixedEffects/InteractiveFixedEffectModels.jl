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
	@show result.rss
	@show result.converged
end
```

Result
```
method : levenberg_marquardt
  7.158471 seconds (17.20 M allocations: 485.516 MB, 4.69% gc time)
result.iterations = 167
result.rss = 1.404830098782482e8
result.converged = true
method : dogleg
  1.018161 seconds (477.79 k allocations: 230.311 MB, 10.54% gc time)
result.iterations = 22
result.rss = 3.733383717783933e8
result.converged = true
method : gauss_seidel
  0.915646 seconds (979 allocations: 146.277 MB, 2.69% gc time)
result.iterations = 18
result.rss = 1.4048300982167023e8
result.converged = true
```

### N x T x 4/5
```julia
for method in [:levenberg_marquardt, :dogleg, :gauss_seidel]
	println("method : $(method)")
	@time result = fit(SparseFactorModel(:id, :time, 1), y ~ 1|> id + time, unbalanceddf, method = method, save = false)
	@show result.iterations
	@show result.rss
	@show result.converged
end
```

Result
```
method : levenberg_marquardt
  5.677054 seconds (17.29 M allocations: 442.471 MB, 4.69% gc time)
result.iterations = 228
result.rss = 1.1234472896726151e8
result.converged = true
method : dogleg
  5.194259 seconds (8.01 M allocations: 300.815 MB, 3.69% gc time)
result.iterations = 237
result.rss = 1.1234472896838553e8
result.converged = true
method : gauss_seidel
  0.675958 seconds (943 allocations: 117.136 MB, 1.59% gc time)
result.iterations = 15
result.rss = 1.123447289365168e8
result.converged = true
```


### N x T x 1/5
```julia
for method in [:levenberg_marquardt, :dogleg, :gauss_seidel]
	println("method : $(method)")
	@time result = fit(SparseFactorModel(:id, :time, 1), y ~ 1|> id + time, sparsedf, method = method, save = false)
	@show result.iterations
	@show result.rss
	@show result.converged
end
```

Result
```
method : levenberg_marquardt
  2.552608 seconds (27.37 M allocations: 463.003 MB, 6.03% gc time)
result.iterations = 378
result.rss = 2.8100350091434512e7
result.converged = true
method : dogleg
  2.202181 seconds (13.23 M allocations: 247.350 MB, 1.55% gc time)
result.iterations = 415
result.rss = 2.810035008876314e7
result.converged = true
method : gauss_seidel
  0.215576 seconds (976 allocations: 29.687 MB, 1.54% gc time)
result.iterations = 19
result.rss = 2.810035006800005e7
result.converged = true
```



# interactive fixed effect 

### N x T

```julia
for method in [:levenberg_marquardt, :dogleg, :gauss_seidel, :regar]
	println("method : $(method)")
	@time result = fit(SparseFactorModel(:id, :time, 2), y ~ x1 |> id + time, df, method = method, save = false)
	@show result.iterations
	@show result.rss
	@show result.converged
end
```

```
method : levenberg_marquardt
 19.741259 seconds (58.97 M allocations: 1.621 GB, 3.10% gc time)
result.iterations = 371
result.rss = 2.4806926953510325e6
result.converged = true
method : dogleg
INFO: Algorithm ended up on a local minimum. Restarting from a new, random, x0.
 12.594733 seconds (12.63 M allocations: 1.174 GB, 5.74% gc time)
result.iterations = 212
result.rss = 2.4806926953499895e6
result.converged = true
method : gauss_seidel
 8.166197 seconds (706.90 k allocations: 750.712 MB, 4.39% gc time)
result.iterations = 57
result.rss = 2.4806927020173306e6
result.converged = true
method : regar
  8.466838 seconds (7.89 M allocations: 976.448 MB, 6.37% gc time)
result.iterations = 187
result.rss = 2.4806926953436537e6
result.converged = true
```

### N x T x 4/5


```julia
for method in [:levenberg_marquardt, :dogleg, :gauss_seidel, :regar]
	println("method : $(method)")
	@time result = fit(SparseFactorModel(:id, :time, 2), y ~ x1 |> id + time, unbalanceddf, method = method, save = false)
	@show result.iterations
	@show result.rss
	@show result.converged
end
```

```
method : levenberg_marquardt
 15.968084 seconds (55.60 M allocations: 1.423 GB, 3.91% gc time)
result.iterations = 365
result.rss = 1.997676617351034e6
result.converged = true
method : dogleg
 33.113829 seconds (59.65 M allocations: 1.483 GB, 1.68% gc time)
result.iterations = 1104
result.rss = 1.9976766273685207e6
result.converged = true
method : gauss_seidel
  9.755387 seconds (708.27 k allocations: 603.140 MB, 3.70% gc time)
result.iterations = 81
result.rss = 1.9986716369475075e6
result.converged = true
method : regar
 29.806383 seconds (54.87 M allocations: 1.352 GB, 1.55% gc time)
result.iterations = 1515
result.rss = 1.9976766236881565e6
result.converged = true
```

### N x T x 1/5


```julia
for method in [:levenberg_marquardt, :dogleg, :gauss_seidel, :regar]
	println("method : $(method)")
	@time result = fit(SparseFactorModel(:id, :time, 2), y ~ x1 |> id + time, sparsedf, method = method, save = false)
	@show result.iterations
	@show result.rss
	@show result.converged
end
```

```
method : levenberg_marquardt
  5.169143 seconds (55.01 M allocations: 993.576 MB, 4.67% gc time)
result.iterations = 358
result.rss = 542208.191393264
result.converged = true
method : dogleg
  9.782926 seconds (66.56 M allocations: 1.143 GB, 2.40% gc time)
result.iterations = 1226
result.rss = 542208.1914324684
result.converged = true
method : gauss_seidel
  1.180293 seconds (704.51 k allocations: 160.071 MB, 7.41% gc time)
result.iterations = 15
result.rss = 543565.5889823908
result.converged = true
method : regar
 17.250520 seconds (114.28 M allocations: 1.838 GB, 1.57% gc time)
result.iterations = 3589
result.rss = 542208.191398862
result.converged = true
```
