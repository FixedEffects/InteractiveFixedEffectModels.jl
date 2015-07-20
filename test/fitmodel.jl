using RDatasets, DataFrames, PanelFactorModels, Base.Test

df = dataset("plm", "Cigar")
df[:pState] = pool(df[:State])
df[:pYear] = pool(df[:Year])

result = fit(PanelFactorModel(:pState, :pYear, 2), Sales ~ Price, df, tol = 1e-9, maxiter = 10000)

# test coef
@test_approx_eq_eps result.coef [163.01350,-0.40610] 1e-4
@test_approx_eq_eps result.factors[1]  -1.248038 1e-4
@test_approx_eq_eps result.loadings[1]  33.62829 1e-4

# check normalization F'F/T = Id and Lambda' Lambda = diag
@test_approx_eq transpose(result.factors) * result.factors  30 * eye(size(result.factors, 2))
@test_approx_eq_eps (transpose(result.loadings)* result.loadings - diagm(diag(transpose(result.loadings)* result.loadings))) fill(zero(Float64), (size(result.loadings, 2), size(result.loadings, 2))) 1e-8


# test coef with absorb option
@test_approx_eq_eps fit(PanelFactorModel(:pState, :pYear, 2),  Sales ~ Price |> pState, df , tol = 1e-9).coef  [-0.42538] 1e-4


