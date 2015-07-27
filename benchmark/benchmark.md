using DataFrames, PanelFactorModels
M = 10000
N = 1000
T = 1000
id =  pool(rand(1:N, M))
time =  pool(rand(1:T, M))
l1 = PooledDataArray(DataArrays.RefArray(id.refs), randn(N))
f1 = PooledDataArray(DataArrays.RefArray(time.refs), randn(T))
l2 = PooledDataArray(DataArrays.RefArray(id.refs), randn(N))
f2 = PooledDataArray(DataArrays.RefArray(time.refs), randn(T))
x1 = Array(Float64, M)
y = similar(x1)
for i in 1:M
  x1[i] = f1[id[i]] * l1[time[i]] + f2[id[i]] * l2[time[i]] + randn()
  y[i] = 3 * x1[i] + 4 * f1[id[i]] * l1[time[i]] + f2[id[i]] * l2[time[i]] + randn()
end
df = DataFrame(id = id, time = time, x1 = x1, y = y)
for method in [:bfgs, :l_bfgs, :svd, :gs, :gradient_descent]
println("$method : factor model")
  @time fit(PanelFactorModel(:id, :time, 1), y ~ 1 |> id, df, tol = 1e-3, method = method) 
  @time fit(PanelFactorModel(:id, :time, 2), y ~ 1 |> id, df, tol = 1e-3, method = method)  
  println("$method : linear factor model")
  @time fit(PanelFactorModel(:id, :time, 1), y ~ x1, df, tol = 1e-3, method = method)  
  @time fit(PanelFactorModel(:id, :time, 2), y ~ x1 |> id, df, tol = 1e-3, method = method) 
end
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
svd : factor model
elapsed time: 21.116349975 seconds (1294930204 bytes allocated, 6.38% gc time)
elapsed time: 26.209531948 seconds (1577580784 bytes allocated, 6.41% gc time)
svd : linear factor model
elapsed time: 43.894730019 seconds (2659773264 bytes allocated, 6.32% gc time)
elapsed time: 33.014362555 seconds (2005976120 bytes allocated, 6.30% gc time)
gs : factor model
elapsed time: 1.06153739 seconds (5822840 bytes allocated)
elapsed time: 1.965853748 seconds (3272816 bytes allocated)
gs : linear factor model
elapsed time: 1.099704219 seconds (5228384 bytes allocated)
elapsed time: 2.01334923 seconds (4573248 bytes allocated)
gradient_descent : factor model
elapsed time: 0.083974254 seconds (6164620 bytes allocated)
elapsed time: 0.072009936 seconds (10638216 bytes allocated, 73.50% gc time)
gradient_descent : linear factor model
elapsed time: 8.506200819 seconds (1529276516 bytes allocated, 19.79% gc time)
elapsed time: 13.170571735 seconds (3131808856 bytes allocated, 26.14% gc time)


