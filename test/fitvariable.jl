using RDatasets, DataFrames, PanelFactorModels, Base.Test
df = dataset("plm", "Cigar")
df[:pState] =  pool(df[:State])
df[:pYear] =  pool(df[:Year])
result = fit(PanelFactorModel(:pState, :pYear, 2), :Sales, df)
@test_approx_eq_eps result.loadings[1, :]  [587.227,  64.3341] 1e-3

resultw = fit(PanelFactorModel(:pState, :pYear, 2), :Sales, df, weight = :Pop)
@test_approx_eq_eps resultw.loadings[1, :]  [ 591.743, 582.089] 1e-3



resultr = fit(PanelFactorModel(:pState, :pYear, 2), :Sales, df, lambda = 400.0)
@test norm(resultr.loadings, 2) + norm(resultr.factors, 2) < norm(result.loadings, 2) + norm(result.factors, 2)