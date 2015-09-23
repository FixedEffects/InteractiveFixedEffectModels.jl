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
for method in [:ar, :lm, :dl]
	println("method : $(method)")
	@time result = fit(SparseFactorModel(:id, :time, 2), y ~ x1 |> id + time, df, method = method, save = false)
	@show result.iterations
	@show result.ess
	@show result.converged
end
```

```
method : ar
 11.239175 seconds (529.78 k allocations: 495.686 MB, 2.98% gc time)
result.iterations = 112
result.ess = 2.480692695343655e6
result.converged = true
method : lm
 28.779354 seconds (67.32 M allocations: 1.497 GB, 2.00% gc time)
result.iterations = 246
result.ess = 2.480692695343654e6
result.converged = true
method : dl
 30.524696 seconds (26.62 M allocations: 902.534 MB, 1.19% gc time)
result.iterations = 234
result.ess = 2.480692695343654e6
result.converged = true
```

### N x T x 4/5


```julia
for method in [:ar, :lm, :dl]
	println("method : $(method)")
	@time result = fit(SparseFactorModel(:id, :time, 2), y ~ x1 |> id + time, unbalanceddf, method = method, maxiter = 10_000, save = false)
	@show result.iterations
	@show result.ess
	@show result.converged
end
```

```
method : ar
 14.164756 seconds (6.83 k allocations: 378.363 MB, 1.65% gc time)
result.iterations = 184
result.ess = 1.9986716476135303e6
result.converged = true
method : lm
 22.679409 seconds (66.03 M allocations: 1.369 GB, 1.78% gc time)
result.iterations = 243
result.ess = 1.997676617223449e6
result.converged = true
method : dl
 38.026766 seconds (46.86 M allocations: 1.083 GB, 0.83% gc time)
result.iterations = 421
result.ess = 1.997676617223449e6
result.converged = true
```

### N x T x 1/5


```julia
for method in [:ar, :lm, :dl]
	println("method : $(method)")
	@time result = fit(SparseFactorModel(:id, :time, 2), y ~ x1 |> id + time, sparsedf, method = method, maxiter = 10_000, save = false)
	@show result.iterations
	@show result.ess
	@show result.converged
end
```

```
method : ar
  1.177007 seconds (2.64 k allocations: 95.489 MB, 1.34% gc time)
result.iterations = 36
result.ess = 543565.5890788117
result.converged = true
method : lm
 14.943738 seconds (156.99 M allocations: 2.437 GB, 2.57% gc time)
result.iterations = 578
result.ess = 551660.8545273632
result.converged = true
method : dl
 13.430610 seconds (64.08 M allocations: 1.053 GB, 1.03% gc time)
result.iterations = 575
result.ess = 551660.8545273632
result.converged = true
```
