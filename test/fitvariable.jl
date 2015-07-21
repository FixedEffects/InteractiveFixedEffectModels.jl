using RDatasets, DataFrames, PanelFactorModels, Base.Test
df = dataset("plm", "Cigar")
df[:pState] =  pool(df[:State])
df[:pYear] =  pool(df[:Year])
@test_approx_eq_eps fit(PanelFactorModel(:pState, :pYear, 2), :Sales, df).loadings[1, : ]  [107.213, 11.7453] 1e-3
@test_approx_eq_eps fit(PanelFactorModel(:pState, :pYear, 2), :Sales, df, weight = :Pop).loadings[1, : ]  [108.037, 12.1665] 1e-3


result = fit(PanelFactorModel(:pState, :pYear, 2), :Sales, df, weight = :Pop, lambda = 0.0)
result_ridge = fit(PanelFactorModel(:pState, :pYear, 2), :Sales, df, weight = :Pop, lambda = 400.0)
@test norm(result_ridge.loadings, 2) + norm(result_ridge.factors, 2) < norm(result.loadings, 2) + norm(result.factors, 2)