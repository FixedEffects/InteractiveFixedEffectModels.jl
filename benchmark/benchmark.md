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


@time result = fit(SparseFactorModel(:id, :time, 2), y ~ x1 |> id + time, df, method = :gs, maxiter = 1_000_000)
@time result = fit(SparseFactorModel(:id, :time, 2), y ~ x1 |> id + time, df, method = :gradient_descent, maxiter = 1_000_000)
@time result = fit(SparseFactorModel(:id, :time, 2), y ~ x1 |> id + time, df, method = :svd, maxiter = 1_000_000)
@time result = fit(SparseFactorModel(:id, :time, 2), y ~ x1 |> id + time, df, method = :bfgs, maxiter = 1_000_000)
@time result = fit(SparseFactorModel(:id, :time, 2), y ~ x1 |> id + time, df, method = :l_bfgs, maxiter = 1_000_000)
```


Results:
```julia
# elapsed time: 2.616350412 seconds (529388596 bytes allocated, 22.96% gc time)
# elapsed time: 23.466209663 seconds (7161736404 bytes allocated, 33.18% gc time)
```


