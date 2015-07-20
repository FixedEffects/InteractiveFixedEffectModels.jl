
This package estimates factor models on datasets in a "panel" form, where each row represents a pair id x time. Compared to the matrix N x T approach, a panel data is generally easier to work with and more memory efficient than a matrix N x T

Construct an object of type `PanelFactorModel`, 

```julia
using RDatasets, DataFrames, FixedEffectModels
df = dataset("plm", "Cigar")
df[:pState] =  pool(df[:State])
df[:pYear] =  pool(df[:Year])
pfm = PanelFactorModel(:pState, :pYear, 2)
```

#### Estimate factor models by incremental SVD
Estimate a factor model for the variable `Sales`

```julia
fit(pfm, :Sales, df)
fit(pfm, :Sales, df, weight = :Pop)
```

The factor model is estimated by incremental SVD, i.e. by minimizing the sum of the squared residuals incrementally for each dimension. By default, the minimization uses a gradient descent. There are three main benefits compared to the usual estimation of a factor model through PCA, this 
- estimate unbalanced panels (with missing (id x time) observation). Another way to do it would be through an EM algorithm, which replaces missing values by the predicted values from the factor model until convergence. This algorithm is generally slower to converge.
- estimate weighted factor models, where weights are not constant within id or time
- avoid the creation of a matrix N x T which may be huge (as in the Netflix problem).


#### Interactive Fixed Effect Models
Estimate models with interactive fixed effects (Bai 2009) 

```julia
fit(pfm, Sales ~ Price, df)
fit(pfm, Sales ~ Price |> pState + pYear, df)
```