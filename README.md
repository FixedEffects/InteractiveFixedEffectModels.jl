
## Motivation

This package estimates linear factor models on "long" datasets, where each row represents an outcome for a pair id x time. The dataset does not have to contain a row for each pair id x time, which allows to fit factor models on severely unbalanced panels (as in the Netflix problem).

For an observation `i`, denote `jλ(i)` its id and `jf(i)` its time.  This package estimates the set of coefficients `beta`, of factors `(f1, .., fr)` and of loadings `(λ1, ..., λr)` that solve

![minimization](img/minimization.png)


When X is a set of id or time dummies, this corresponds to a PCA with missing values. When X is a general set of regressors, this corresponds to a model with interactive fixed effects as described in Bai (2009).


## Syntax

The general syntax is
```julia
fit(pfm::PanelFactorModel,
	f::Formula, 
    df::AbstractDataFrame, 
    vcov_method::AbstractVcovMethod = VcovSimple(),
 	method::Symbol
    weight::Union(Symbol, Nothing) = nothing, 
    subset::Union(AbstractVector{Bool}, Nothing) = nothing, 
    maxiter::Int64 = 10000, tol::Float64 = 1e-8
    )
```


- The first argument of `fit` is an object of type `PanelFactorModel`. Such an object can be constructed by specifying the id variable, the time variable, and the rank of the factor model. Both the id and time variable must be of type `PooledDataVector`.

	```julia
	using RDatasets, DataFrames, PanelFactorModels
	df = dataset("plm", "Cigar")
	# create PooledDataVector
	df[:pState] =  pool(df[:State])
	df[:pYear] =  pool(df[:Year])
	# create PanelFactorModel in state, year, and rank 2
	factormodel = PanelFactorModel(:pState, :pYear, 2)
	```

- The second argument of `fit` is a formula
	- When the only regressor is `0`, `fit` fits a factor model on the left hand side variable

		```julia
		fit(PanelFactorModel(:pState, :pYear, 2), Sales ~ 0)
		```


		You can pre-demean the variable using `|>` as in the package [FixedEffectModels.jl](https://github.com/matthieugomez/FixedEffectModels.jl).

		```
		fit(PanelFactorModel(:pState, :pYear, 2), Sales |> pState, df)
		fit(PanelFactorModel(:pState, :pYear, 2), Sales |> pYear, df)
		fit(PanelFactorModel(:pState, :pYear, 2), Sales |> pState + pYear, df)
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

- `:gs` (default) This method fits a factor model by alternating regressions on loadings interacted with time dummy and factors interacted by id dummy, as in the [Gauss-Seidel method](https://en.wikipedia.org/wiki/Gauss%E2%80%93Seidel_method). In my tests, this is the fastest and most memory efficient method. 


- `:svd`. This method fits a factor model through the method described in Bai (2009). In case of an unbalanced panel, missing values are replaced by the values predicted by the factor model fitted in the previous iteration. The `:svd` method requires that the initial dataset contains unique observations for a given pair id x time, and that there is enough RAM to store a matrix NxT. The `svd` method is fast when T/N is small and when the number of missing values is small.


- Optimization methods (such as `:gradient_descent`, `:bgfgs` and `:l_bgfs`). These methods estimate the model by directly minimizing the sum of squared residuals using the Package Optim. This can be very fast in some problem.

You may find some speed comparisons [here](benchmark/benchmark.md)

## weights

The `weights` option allows to minimize the sum of weighted residuals. This option is not available for the option `:svd`. 

When weights are not constant within id or time, the optimization problem has local minima that are not global. You may want to use the method `:gs` rather than an optimization method

## lambda
`lambda` adds a Tikhonov regularization term to the sum of squared residuals. This option is only available when using an optimization method.

## save
The option `save = true` saves a new dataframe storing residuals, factors, loadings and the eventual fixed effects. Importantly, the new dataframe is aligned with the initial dataframe (rows not used in the estimation are simply filled with NA).

## Install

```julia
Pkg.clone("https://github.com/matthieugomez/PanelFactorModels.jl")
```

## References
- Bai, Jushan. *Panel data models with interactive fixed effects.* (2009) Econometrica 
- Ilin, Alexander, and Tapani Raiko. *Practical approaches to principal component analysis in the presence of missing values.* (2010) The Journal of Machine Learning Research 11 
-  Koren, Yehuda. *Factorization meets the neighborhood: a multifaceted collaborative filtering model.* (2008) Proceedings of the 14th ACM SIGKDD international conference on Knowledge discovery and data mining. 
- Srebro, Nathan, and Tommi Jaakkola. *Weighted low-rank approximations* (2010) The Journal of Machine Learning Research 11 

