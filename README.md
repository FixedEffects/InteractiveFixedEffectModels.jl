
This package estimates factor models on dataframes, and more generally on datasets in a "panel" form, where each row represents a pair (id, time).
This "panel" form is easier to work with and more memory efficient than a matrix N x T

#### Factor Models
Construct an object of type `PanelFactorModel`, 

```julia
using RDatasets, DataFrames, FixedEffectModels
df = dataset("plm", "Cigar")
df[:pState] =  pool(df[:State])
df[:pYear] =  pool(df[:Year])
pfm = PanelFactorModel(:pState, :pYear, 2)
```


Estimate a factor model for the variable `Sales`

```julia
fit(pfm, :Sales, df)
fit(pfm, :Sales, df, weight = :Pop)
```

The function estimates the factor model by incremental SVD, i.e. minimizing the sum of the squared residuals incrementally for each dimension. By default, the minimization uses a gradient descent. Contrary to the usual estimation of a factor model through PCA
- this allows to estimate unbalanced panels. Another way to do it is the EM algorithm (replace missing values by the predicted values from the factor model until convergence). This solution converges more slowly 
- this allows to estimate weighted PCA, where weights are not constant within id or time
- this avoids to construct a matrix N x T which may be huge (as in the Netflix problem).


#### Interactive Fixed Effect Models
Estimate models with interactive fixed effects (Bai 2009) 

```julia
fit(pfm, Sales ~ Price, df)
fit(pfm, Sales ~ Price |> pState + pYear, df)
```