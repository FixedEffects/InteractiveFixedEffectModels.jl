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
for method in [:ar, :svd, :gd]
	println("method : $(method)")
	@time result = fit(SparseFactorModel(:id, :time, 1), y ~ 1|> id + time, df, method = method, maxiter = 100_000, save = false)
	@show result.ess
	@show result.converged
end
```

Result
```
method : ar
elapsed time: 1.928428095 seconds (173644968 bytes allocated, 6.63% gc time)
result.ess => 1.404830097924866e8
result.converged => Bool[true]
method : svd
elapsed time: 0.485647961 seconds (200014520 bytes allocated, 21.41% gc time)
result.ess => 1.404830097924866e8
result.converged => Bool[true]
method : gd
  3.408362 seconds (1.41 M allocations: 167.936 MB, 2.27% gc time)
result.ess = 1.404830097924866e8
result.converged = Bool[true]
```

### N x T x 4/5
```julia
for method in [:ar, :svd, :gd]
	println("method : $(method)")
	@time result = fit(SparseFactorModel(:id, :time, 1), y ~ 1|> id + time, unbalanceddf, method = method, maxiter = 100_000, save = false)
	@show result.ess
	@show result.converged
end
```

Result
```
method : ar
  1.131804 seconds (829 allocations: 117.179 MB, 5.52% gc time)
result.ess = 1.1234472891128767e8
result.converged = Bool[true]
method : svd
  1.852528 seconds (1.32 k allocations: 221.154 MB, 4.46% gc time)
result.ess = 1.1234472891128775e8
result.converged = Bool[true]
method : gd
  1.793133 seconds (995.87 k allocations: 132.403 MB, 3.76% gc time)
result.ess = 1.1234472891128767e8
result.converged = Bool[true]
```


### N x T x 1/5
```julia
for method in [:ar, :svd, :gd]
	println("method : $(method)")
	@time result = fit(SparseFactorModel(:id, :time, 1), y ~ 1|> id + time, sparsedf, method = method, maxiter = 100_000, save = false)
	@show result.ess
	@show result.converged
end
```

Result
```
method : ar
  0.312489 seconds (835 allocations: 29.729 MB, 1.49% gc time)
result.ess = 2.8100350063354842e7
result.converged = Bool[true]
method : svd
  9.717171 seconds (5.14 k allocations: 365.202 MB, 0.95% gc time)
result.ess = 2.8100350063355103e7
result.converged = Bool[true]
method : gd
  0.608217 seconds (1.32 M allocations: 49.866 MB, 0.61% gc time)
result.ess = 2.8100350063354842e7
result.converged = Bool[true]
```



# interactive fixed effect 

### N x T

```julia
for method in [:ar, :svd, :gd]
	println("method : $(method)")
	@time result = fit(SparseFactorModel(:id, :time, 2), y ~ x1 |> id + time, df, method = method, maxiter = 10000, save = false)
	@show result.ess
	@show result.converged
end
```

```
method : ar
 16.530557 seconds (426.91 k allocations: 597.985 MB, 1.39% gc time)
result.ess = 2.4806926953436653e6
result.converged = true
method : svd
 33.029899 seconds (294.79 k allocations: 6.909 GB, 3.09% gc time)
result.ess = 2.4806926953436537e6
result.converged = true
method : gd
 27.051480 seconds (10.56 M allocations: 749.053 MB, 1.48% gc time)
result.ess = 2.480692695343663e6
result.converged = true
```

### N x T x 4/5


```julia
for method in [:ar, :svd, :gd]
	println("method : $(method)")
	@time result = fit(SparseFactorModel(:id, :time, 2), y ~ x1 |> id + time, unbalanceddf, method = method, maxiter = 10_000, save = false)
	@show result.ess
	@show result.converged
end
```

```
method : ar
 22.695062 seconds (260.21 k allocations: 473.567 MB, 1.01% gc time)
result.ess = 1.9986716476040236e6
result.converged = true
method : svd
 39.367397 seconds (270.46 k allocations: 7.269 GB, 2.27% gc time)
result.ess = 1.9976766172140841e6
result.converged = true
method : gd
 38.854486 seconds (16.82 M allocations: 726.995 MB, 0.65% gc time)
result.ess = 1.9986716476040212e6
result.converged = true
```

### N x T x 1/5


```julia
for method in [:ar, :svd, :gd]
	println("method : $(method)")
	@time result = fit(SparseFactorModel(:id, :time, 2), y ~ x1 |> id + time, sparsedf, method = method, maxiter = 10_000, save = false)
	@show result.ess
	@show result.converged
end
```

```
method : ar
  0.994516 seconds (386.63 k allocations: 124.067 MB, 1.92% gc time)
result.ess = 543565.5890951306
result.converged = true
method : svd
 22.306019 seconds (395.48 k allocations: 1.835 GB, 1.04% gc time)
result.ess = 542208.1913287133
result.converged = true
method : gd
  1.782945 seconds (2.63 M allocations: 158.451 MB, 1.31% gc time)
result.ess = 543565.5890951335
result.converged = true
```
