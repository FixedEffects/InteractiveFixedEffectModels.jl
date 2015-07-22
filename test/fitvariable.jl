using RDatasets, DataFrames, PanelFactorModels, Base.Test
df = dataset("plm", "Cigar")
df[:pState] =  pool(df[:State])
df[:pYear] =  pool(df[:Year])
result = fit(PanelFactorModel(:pState, :pYear, 2), :Sales, df)
result = fit(PanelFactorModel(:pState, :pYear, 2), :Sales, df, method = :newton)




@test_approx_eq_eps result.loadings[1, :]  [107.213, 11.7453] 1e-3
resultw = fit(PanelFactorModel(:pState, :pYear, 2), :Sales, df, weight = :Pop)
@test_approx_eq_eps resultw.loadings[1, :]  [108.037, 12.1665] 1e-3



resultw = fit(PanelFactorModel(:pState, :pYear, 2), :Sales, weight = :Pop, df)
resultw = fit(PanelFactorModel(:pState, :pYear, 2), :Sales, weight = :Pop, df, method = :newton)


result = fit(PanelFactorModel(:pState, :pYear, 2), :Sales, df, weight = :Pop, lambda = 0.0)
result_ridge = fit(PanelFactorModel(:pState, :pYear, 2), :Sales, df, weight = :Pop, lambda = 400.0)
@test norm(result_ridge.loadings, 2) + norm(result_ridge.factors, 2) < norm(result.loadings, 2) + norm(result.factors, 2)