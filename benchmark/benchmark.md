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
for method in [:levenberg_marquardt, :dogleg]
	println("method : $(method)")
	@time result = fit(SparseFactorModel(:id, :time, 1), y ~ 1|> id + time, df, method = method, maxiter = 100_000, save = false)
	@show result.iterations
	@show result.ess
	@show result.converged
end
```

Result
```
method : levenberg_marquardt
  0.901253 seconds (1.41 M allocations: 244.388 MB, 10.53% gc time)
result.iterations = 12
result.ess = 3.7333837192913985e8
result.converged = true
method : dogleg
  0.760176 seconds (330.66 k allocations: 228.066 MB, 12.98% gc time)
result.iterations = 16
result.ess = 3.733383719362705e8
result.converged = true
```

### N x T x 4/5
```julia
for method in [:levenberg_marquardt, :dogleg]
	println("method : $(method)")
	@time result = fit(SparseFactorModel(:id, :time, 1), y ~ 1|> id + time, unbalanceddf, method = method, maxiter = 100_000, save = false)
	@show result.iterations
	@show result.ess
	@show result.converged
end
```

Result
```
method : levenberg_marquardt
  1.559850 seconds (4.41 M allocations: 245.819 MB, 6.27% gc time)
result.iterations = 44
result.ess = 2.986821436455046e8
result.converged = true
method : dogleg
  5.488896 seconds (6.96 M allocations: 284.876 MB, 3.19% gc time)
result.iterations = 286
result.ess = 1.123447295629448e8
result.converged = true
```


### N x T x 1/5
```julia
for method in [:levenberg_marquardt, :dogleg]
	println("method : $(method)")
	@time result = fit(SparseFactorModel(:id, :time, 1), y ~ 1|> id + time, sparsedf, method = method, maxiter = 100_000, save = false)
	@show result.iterations
	@show result.ess
	@show result.converged
end
```

Result
```
method : levenberg_marquardt
  1.963581 seconds (22.06 M allocations: 381.892 MB, 3.12% gc time)
result.iterations = 260
result.ess = 2.8100350265781414e7
result.converged = true
method : dogleg
  1.310983 seconds (5.15 M allocations: 123.958 MB, 8.35% gc time)
result.iterations = 248
result.ess = 2.8100350234330516e7
result.converged = true
```



# interactive fixed effect 

### N x T

```julia
for method in [:levenberg_marquardt, :dogleg]
	println("method : $(method)")
	@time result = fit(SparseFactorModel(:id, :time, 2), y ~ x1 |> id + time, df, method = method, save = false)
	@show result.iterations
	@show result.ess
	@show result.converged
end
```

```
method : levenberg_marquardt
 11.712379 seconds (37.41 M allocations: 1.281 GB, 4.72% gc time)
result.iterations = 245
result.ess = 2.480692695345996e6
result.converged = true
method : dogleg
 11.925921 seconds (16.14 M allocations: 987.386 MB, 4.22% gc time)
result.iterations = 290
result.ess = 2.480692695382266e6
result.converged = true
```

### N x T x 4/5


```julia
for method in [:levenberg_marquardt, :dogleg]
	println("method : $(method)")
	@time result = fit(SparseFactorModel(:id, :time, 2), y ~ x1 |> id + time, unbalanceddf, method = method, maxiter = 10_000, save = false)
	@show result.iterations
	@show result.ess
	@show result.converged
end
```

```
method : levenberg_marquardt
 18.711958 seconds (74.18 M allocations: 1.685 GB, 3.44% gc time)
result.iterations = 483
result.ess = 1.9976766172393495e6
result.converged = true
method : dogleg
 17.717442 seconds (31.69 M allocations: 1.052 GB, 2.71% gc time)
result.iterations = 588
result.ess = 1.9976766172565985e6
result.converged = true
```

### N x T x 1/5


```julia
for method in [:levenberg_marquardt, :dogleg]
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
