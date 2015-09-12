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
for method in [:ar, :svd]
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
```

### N x T x 4/5
```julia
for method in [:ar]
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
```


### N x T x 1/5
```julia
for method in [:ar]
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
```



# interactive fixed effect 

### N x T

```julia
for method in [:ar, :lm]
	println("method : $(method)")
	@time result = fit(SparseFactorModel(:id, :time, 2), y ~ x1 |> id + time, df, method = method, maxiter = 10000, save = false)
	@show result.ess
	@show result.converged
end
```

```
method : ar
 12.117485 seconds (5.77 k allocations: 2.473 GB, 5.22% gc time)
result.ess = 2.4806926953436593e6
result.converged = true
method : lm
 22.340266 seconds (3.41 k allocations: 568.600 MB, 1.08% gc time)
result.ess = 2.480692695344041e6
result.converged = true
```

### N x T x 4/5


```julia
for method in [:ar, :lm]
	println("method : $(method)")
	@time result = fit(SparseFactorModel(:id, :time, 2), y ~ x1 |> id + time, unbalanceddf, method = method, maxiter = 10_000, save = false)
	@show result.ess
	@show result.converged
end
```

```
method : ar
 15.569802 seconds (8.54 k allocations: 2.962 GB, 4.24% gc time)
result.ess = 1.9986716476047342e6
result.converged = true
method : lm
 14.661392 seconds (3.42 k allocations: 455.318 MB, 1.14% gc time)
result.ess = 1.9976766172148217e6
result.converged = true
```

### N x T x 1/5


```julia
for method in [:ar, :lm]
	println("method : $(method)")
	@time result = fit(SparseFactorModel(:id, :time, 2), y ~ x1 |> id + time, sparsedf, method = method, maxiter = 10_000, save = false)
	@show result.ess
	@show result.converged
end
```

```
method : ar
 22.953487 seconds (13.33 k allocations: 755.468 MB, 0.47% gc time)
result.ess = 554512.0500473888
result.converged = true
method : lm
 24.461941 seconds (5.03 k allocations: 115.327 MB, 0.29% gc time)
result.ess = 542208.1913295744
result.converged = true
```
