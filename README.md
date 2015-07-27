
## Motivation

This package fits models of the form
<img src="img/model.png" alt="Model" width = "716" height = "152">
The estimate are obtained by solving the following optimization problem
<img src="img/minimization.png" alt="minimization"  width = "1079" height = "162">

Traditional estimation of factor models  requires a matrix N x T and that the set of regressors is null or equals a set of id or time dummies. In contrast, 

- This package estimates factor models on "long" datasets, where each row represents an outcome for a pair id x time. In particular, there may be zero or more than one observed outcome per pair. This allows to fit factor models on severely unbalanced panels (as in the Netflix problem).
-  X can be any set of regressors. Estimation of this general model is described in Bai (2009). 



## Syntax

The general syntax is
```julia
fit(pfm::PanelFactorModel,
	f::Formula, 
    df::AbstractDataFrame, 
 	method::Symbol
    weight::Union(Symbol, Nothing) = nothing, 
    subset::Union(AbstractVector{Bool}, Nothing) = nothing, 
    maxiter::Int64 = 10000, tol::Float64 = 1e-8
    )
```


- The first argument of `fit` is an object of type `PanelFactorModel`. Such an object can be constructed by specifying the id variable, the time variable, and the factor dimension in the dataframe. Both the id and time variable must be of type `PooledDataVector`.

	```julia
	using RDatasets, DataFrames, PanelFactorModels
	df = dataset("plm", "Cigar")
	# create PooledDataVector
	df[:pState] =  pool(df[:State])
	df[:pYear] =  pool(df[:Year])
	# create PanelFactorModel in state, year, and rank 2
	factormodel = PanelFactorModel(:pState, :pYear, 2)
	```

- The fit argument of `fit` is either a symbol or a formula
	- When the only regressor is `1`, it simply fits a factor model on the left hand side variable

		```julia
		fit(Sales ~ 1, PanelFactorModel(:pState, :pYear, 2))
		```

		Fit a factor variable on a demeaned variable using `|>` as in the package [FixedEffectModels.jl](https://github.com/matthieugomez/FixedEffectModels.jl).

		```
		fit(PanelFactorModel(:pState, :pYear, 2), Sales ~ Price |> pState, df)
		fit(PanelFactorModel(:pState, :pYear, 2), Sales ~ Price |> pYear, df)
		fit(PanelFactorModel(:pState, :pYear, 2), Sales ~ Price |> pState + pYear, df)
		```

	- With multiple regressors, `fit` fits a linear model with interactive fixed effects (Bai (2009))
	

		```julia
		fit(PanelFactorModel(:pState, :pYear, 2), Sales ~ Price, df)
		```

		Similarly, you may add id  or time fixed effects
		```julia
		fit(PanelFactorModel(:pState, :pYear, 2), Sales ~ Price |> pState, df)
		```


## method
Three methods are available

- `:gs` (default) This option fits a factor model by alternating regressions on loadings interacted with time dummy and factors interacted by id dummy, as in the [gauss-seidel iterations](https://en.wikipedia.org/wiki/Gauss%E2%80%93Seidel_method). This is generally the fastest and most memory efficient method. 


- `:svd`. This option fits a factor model through the method described in Bai (2009). In case of an unbalanced panel, missing values are replaced by the values predicted by the factor model fitted in the previous iteration. 

The `:svd` method requires that the initial dataset contains unique observations for a given pair id x time, and that there is enough RAM to store a matrix NxT. The `svd` method is fast when T/N is small and when the number of missing values is small.


- Optimization methods (such as `:gradient_descent`, `:bgfgs` and `:l_bgfs`). These methods estimate the model by directly minimizing the sum of squared residuals using the Package Optim. This can be very fast in some problem.

You may find some comparisons [here](benchmar/benchmark.md)

## weights

The `weights` option allows to minimize the sum of weighted residuals. This option is not available for the option `:svd`. 

When weights are not constant within id or time, the optimization problem has local minima that are not global. You mawy want to use the method `:gs` rather than an optimization method

## lambda
`lambda` adds a Tikhonov regularization term to the sum of squared residuals, i.e.

	```
	sum of residuals +  lambda( ||factors||^2 + ||loadings||^2)
	```
This option is only available for optimziation methods

## save
The option `save = true` saves a new dataframe which stores residuals, factors, loadings and the eventual fixed effects. The new dataframe is aligned with the initial dataframe: rows not used in the estimation are simply filled with NA.

## Install

```julia
Pkg.clone("https://github.com/matthieugomez/PanelFactorModels.jl")
```

## References
- Bai, Jushan. *Panel data models with interactive fixed effects.* (2009) Econometrica 
- Ilin, Alexander, and Tapani Raiko. *Practical approaches to principal component analysis in the presence of missing values.* (2010) The Journal of Machine Learning Research 11 
-  Koren, Yehuda. *Factorization meets the neighborhood: a multifaceted collaborative filtering model.* (2008) Proceedings of the 14th ACM SIGKDD international conference on Knowledge discovery and data mining. 
- Srebro, Nathan, and Tommi Jaakkola. *Weighted low-rank approximations* (2010) The Journal of Machine Learning Research 11 

