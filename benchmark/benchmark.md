```julia
using DataFrames, PanelFactorModels
# compared to the Netflix dataset
# 1/100th of users, 1/100th movies, 1/10000th ratings
M = 10_000
N = 5_000
T = 200
l1 = randn(N)
l2 = randn(N)
f1 = randn(T)
f2 = randn(T)
id =  pool(rand(1:N, M))
time =  pool(rand(1:T, M))
x1 = Array(Float64, M)
y = Array(Float64, M)
for i in 1:M
  x1[i] = f1[time[i]] * l1[id[i]] + f2[time[i]] * l2[id[i]] + l1[id[i]]^2 + f1[time[i]]^2 + randn()
  y[i] = 3 * x1[i] + 4 * f1[time[i]] * l1[id[i]] + f2[time[i]] * l2[id[i]] + l1[id[i]]^2 + f1[time[i]]^2 + randn()
end
df = DataFrame(id = id, time = time, x1 = x1, y = y)
for method in [:gs, :svd, :gradient_descent, :bfgs, :l_bfgs]
	println("\n $method : factor model")
  @time result = fit(PanelFactorModel(:id, :time, 2), x1 ~ 1 |> id + time, df, method = method, maxiter = 1_000_000);
  @show result.converged  
  println("$method : linear factor model")
  @time result = fit(PanelFactorModel(:id, :time, 2), y ~ x1 |> id + time, df, method = method, maxiter = 1_000_000);
  @show result
end
```

Results

```
gs : factor model
elapsed time: 1.06153739 seconds (5822840 bytes allocated)
elapsed time: 1.965853748 seconds (3272816 bytes allocated)
gs : linear factor model
elapsed time: 1.099704219 seconds (5228384 bytes allocated)
elapsed time: 2.01334923 seconds (4573248 bytes allocated)

svd : factor model
elapsed time: 21.116349975 seconds (1294930204 bytes allocated, 6.38% gc time)
elapsed time: 26.209531948 seconds (1577580784 bytes allocated, 6.41% gc time)
svd : linear factor model
elapsed time: 43.894730019 seconds (2659773264 bytes allocated, 6.32% gc time)
elapsed time: 33.014362555 seconds (2005976120 bytes allocated, 6.30% gc time)

gradient_descent : factor model
elapsed time: 0.083974254 seconds (6164620 bytes allocated)
elapsed time: 0.072009936 seconds (10638216 bytes allocated, 73.50% gc time)
gradient_descent : linear factor model
elapsed time: 8.506200819 seconds (1529276516 bytes allocated, 19.79% gc time)
elapsed time: 13.170571735 seconds (3131808856 bytes allocated, 26.14% gc time)

bfgs : factor model
elapsed time: 0.495285785 seconds (99838012 bytes allocated)
elapsed time: 0.700387654 seconds (196365136 bytes allocated, 9.45% gc time)
bfgs : linear factor model
elapsed time: 440.036125888 seconds (1082403940 bytes allocated, 0.24% gc time)
elapsed time: 2102.579752999 seconds (2427278272 bytes allocated, 0.11% gc time)

l_bfgs : factor model
elapsed time: 8.56061226 seconds (4111354596 bytes allocated, 54.26% gc time)
elapsed time: 16.929214625 seconds (8213524848 bytes allocated, 54.97% gc time)
l_bfgs : linear factor model
elapsed time: 84.592223574 seconds (41066602908 bytes allocated, 55.14% gc time)
elapsed time: 168.623512958 seconds (82133944616 bytes allocated, 55.30% gc time)
``

