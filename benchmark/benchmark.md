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
for method in [:ar, :svd]
	println("method : $(method)")
	@time result = fit(SparseFactorModel(:id, :time, 1), y ~ 1|> id + time, unbalanceddf, method = method, maxiter = 100_000, save = false)
	@show result.ess
	@show result.converged
end
```

Result
```
method : ar
elapsed time: 1.400765669 seconds (139027384 bytes allocated, 7.71% gc time)
result.ess => 1.1234472891128767e8
result.converged => Bool[true]
method : svd
elapsed time: 1.494938743 seconds (199710424 bytes allocated, 6.92% gc time)
result.ess => 1.123447289112877e8
result.converged => Bool[true]
```


### N x T x 1/5
```julia
for method in [:ar, :svd]
	println("method : $(method)")
	@time result = fit(SparseFactorModel(:id, :time, 1), y ~ 1|> id + time, sparsedf, method = method, maxiter = 100_000, save = false)
	@show result.ess
	@show result.converged
end
```

Result
```
method : ar
elapsed time: 0.375325981 seconds (35140440 bytes allocated)
result.ess => 2.8100350063354842e7
result.converged => Bool[true]
method : svd
elapsed time: 23.714161639 seconds (733597632 bytes allocated, 3.00% gc time)
result.ess => 2.8100350063354842e7
result.converged => Bool[true]
```



# interactive fixed effect 

### N x T

```julia
for method in [:ar, :svd]
	println("method : $(method)")
	@time result = fit(SparseFactorModel(:id, :time, 2), y ~ x1 |> id + time, df, method = method, maxiter = 10_000, save = false)
	@show result.ess
	@show result.converged
end
```

```
method : ar
elapsed time: 36.882372434 seconds (745731504 bytes allocated, 0.58% gc time)
result.ess => 2.480692695343654e6
result.converged => true
method : svd
elapsed time: 40.361390944 seconds (7574901724 bytes allocated, 17.18% gc time)
result.ess => 2.480692695343654e6
result.converged => true
```

### N x T x 4/5


```julia
for method in [:ar, :svd]
	println("method : $(method)")
	@time result = fit(SparseFactorModel(:id, :time, 2), y ~ x1 |> id + time, unbalanceddf, method = method, maxiter = 10_000, save = false)
	@show result.ess
	@show result.converged
end
```

```
method : ar
elapsed time: 50.933467702 seconds (636024292 bytes allocated, 0.50% gc time)
result.ess => 1.9986716476040047e6
result.converged => true
method : svd
elapsed time: 49.615828724 seconds (7857778020 bytes allocated, 14.30% gc time)
result.ess => 1.9976766172140844e6
result.converged => true
```

### N x T x 1/5


```julia
for method in [:ar, :svd]
	println("method : $(method)")
	@time result = fit(SparseFactorModel(:id, :time, 2), y ~ x1 |> id + time, sparsedf, method = method, maxiter = 10_000, save = false)
	@show result.ess
	@show result.converged
end
```

```
method : ar
elapsed time: 0.98190777 seconds (131943712 bytes allocated)
result.ess => 543565.5890951304
result.converged => true
method : svd
elapsed time: 25.646084433 seconds (2026914288 bytes allocated, 8.07% gc time)
result.ess => 542208.1913287133
result.converged => true

```
