```julia
using DataFrames, SparseFactorModels
# compared to the Netflix dataset
# 1/100th of users, 1/100th movies, 1/10000th ratings
M = 20_000
N = 5_000
T = 200
l1 = randn(N)*T
l2 = randn(N)*T
f1 = randn(T)/T
f2 = randn(T)/T
id =  pool(rand(1:N, M))
time =  pool(rand(1:T, M))
x1 = Array(Float64, M)
y = Array(Float64, M)
for i in 1:M
  x1[i] = 4 + f1[time[i]] * l1[id[i]] + f2[time[i]] * l2[id[i]] + l1[id[i]]^2 + f1[time[i]]^2 + randn()
  y[i] = 5 + 3 * x1[i] + 4 * f1[time[i]] * l1[id[i]] + f2[time[i]] * l2[id[i]] + l1[id[i]]^2 + f1[time[i]]^2 + randn()
end
df = DataFrame(id = id, time = time, x1 = x1, y = y)
for method in [:gs, :gradient_descent, :cg, :momentum_gradient_descent]
	println("method : $(method)")
	@time result = fit(SparseFactorModel(:id, :time, 2), y ~ 1|> id + time, df, method = method, maxiter = 100_000)
	@show result.ess
	@show result.converged
end

for method in [:gs, :gradient_descent, :cg, :momentum_gradient_descent]
	println("method : $(method)")
	@time result = fit(SparseFactorModel(:id, :time, 2), y ~ x1 |> id + time, df, method = method, maxiter = 1_000_000)
	@show result.ess
	@show result.converged
end
```


Result:
```julia
method : gs
elapsed time: 102.944286076 seconds (28768884 bytes allocated)
result.ess => 1.1361913076948249e29
result.converged => Bool[false,false]
method : gradient_descent
elapsed time: 225.952355722 seconds (80264502484 bytes allocated, 37.11% gc time)
result.ess => 272628.24289865873
result.converged => Bool[false,false]
method : cg
elapsed time: 229.711184361 seconds (96712741332 bytes allocated, 43.20% gc time)
result.ess => 317421.22279697633
result.converged => Bool[false,false]
method : momentum_gradient_descent
elapsed time: 73.835700692 seconds (28171286236 bytes allocated, 39.92% gc time)
result.ess => 270399.73489909375
result.converged => Bool[true,true]

julia> for method in [:gs, :gradient_descent, :cg, :momentum_gradient_descent]
       	println("method : $(method)")
       	@time result = fit(SparseFactorModel(:id, :time, 2), y ~ x1 |> id + time, df, method = method, maxiter = 1_000_000)
       	@show result.ess
       	@show result.converged
       end
method : gs
elapsed time: 98.175544643 seconds (75646652 bytes allocated, 0.09% gc time)
result.ess => 45682.29609309232
result.converged => true
method : gradient_descent
elapsed time: 3080.987150135 seconds (810010791132 bytes allocated, 27.37% gc time)
result.ess => 164776.1296901894
result.converged => false
method : cg
elapsed time: 2643.245567594 seconds (973951407628 bytes allocated, 37.70% gc time)
result.ess => 710096.5296445453
result.converged => false
method : momentum_gradient_descent
elapsed time: 2997.246287493 seconds (1000417396284 bytes allocated, 34.74% gc time)
result.ess => 200592.22194620976
result.converged => true
```