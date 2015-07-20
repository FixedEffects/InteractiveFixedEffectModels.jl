
This package estimates factor models on datasets where each row represents an observation.

I'll use the term "panels" to refer to these long datasets, and id x time to refer to the two dimensions of the factor structure - alternatively, they are referred as variable x observation for PCA or user x movie for recommandation systems.

### PanelFactorModel
Starting from a a dataframe,  construct an object of type `PanelFactorModel` by specifying the id variable, the time variable, and the factor dimension. Both the the id and time variable must be of type `PooledDataVector`.

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

The factor model is estimated by incremental SVD, i.e. by minimizing the sum of the squared residuals incrementally for each dimension. By default, the minimization uses a gradient descent. This allows three importants benefits compared to the usual estimation of a factor model through eigenvalue decomposition:
- estimate unbalanced panels (with missing (id x time) observation). Another way to do it would be through an EM algorithm, which replaces missing values by the predicted values from the factor model until convergence. This algorithm is generally slower to converge and takes more memory
- estimate weighted factor models, where weights are not constant within id or time
- avoid the creation of a matrix N x T which takes memory


#### Interactive Fixed Effect Models
Estimate models with interactive fixed effects (Bai 2009) 

```julia
fit(pfm, Sales ~ Price, df)
fit(pfm, Sales ~ Price |> pState + pYear, df)
```


## Install

```julia
Pkg.clone("https://github.com/matthieugomez/PanelFactorModels.jl")
```