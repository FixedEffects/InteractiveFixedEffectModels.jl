
## Motivation
This package estimates factor models on "long" datasets, where each row represents an observation. 

I'll use the term "panels" to refer to these long datasets, and id x time to refer to the two dimensions of the factor structure - corresponding to the pair variable x observation in PCA and the pair user x movie in recommandation problems.



The idea is to directly minimize the sum of residuals. This yields four main benefits compared to a traditional eigenvalue decomposition:

1. estimate unbalanced panels, i.e. with missing (id x time) observations. 
	An alternative would be the EM algorithm, which replaces iteratively missing values by the predicted values from the factor model until convergence. In my experience however, the EM algorithm is generally slower to converge.


2. estimate weighted factor models, where weights are not constant within id or time

3. estimate factor models with a penalization for the norm of loadings and factors (Tikhonov regularization), ie minimizing 

   ```
   sum of squared residuals + lambda *(||loadings||^2 + ||factors||^2)
   ```



## Syntax
- The first argument of `fit` is an object of type `PanelFactorModel`. Such an object `PanelFactorModel` can be constructed by specifying the id variable, the time variable, and the factor dimension in the dataframe. Both the id and time variable must be of type `PooledDataVector`.

	```julia
	using RDatasets, DataFrames, PanelFactorModels
	df = dataset("plm", "Cigar")
	# create PooledDataVector
	df[:pState] =  pool(df[:State])
	df[:pYear] =  pool(df[:Year])
	# create PanelFactorModel in state, year, and rank 2
	factormodel = PanelFactorModel(:pState, :pYear, 2)
	```

- The second argument of `fit` is either a symbol or a formule.
	- When the second argument is a symbol,` fit` fits a factor model on a variable. 

		```julia
		fit(PanelFactorModel(:pState, :pYear, 2), :Sales)
		```
		The factor model is fits iteratively for each dimension.
	- When the second argument is a formula, `fit` fits a linear model with interactive fixed effects (Bai (2009))
	

		```julia
		fit(PanelFactorModel(:pState, :pYear, 2), Sales ~ Price, df)
		```
		Specify id or time fixed effects using `|>` as in the package [FixedEffectModels.jl](https://github.com/matthieugomez/FixedEffectModels.jl).

		```
		fit(PanelFactorModel(:pState, :pYear, 2), Sales ~ Price |> pState + pYear, df)
		```


- `fit` has also keyword arguments:
	- `subset`: estimate the model on a subset of the dataframe
	- `weights`: minimizes the sum of weighted residuals
	- `lambda`: add a Tikhonov regularization, i.e. minimize the 
		```
		sum of residuals +  lambda( ||factors||^2 + ||loadings||^2)
		```
	- `method`: this option is passed to the minimization method from Optim. It defaults to `:gradient_descent` when estimating a factor model, and `:bfgs`  when estimating a linear model with interactive fixed effects.  
	

		Rather than an optimization method, you can also choose the method described in Bai (2009) using `method = :svd`.

## Install

```julia
Pkg.clone("https://github.com/matthieugomez/PanelFactorModels.jl")
```

## References
- Ilin, Alexander, and Tapani Raiko. *Practical approaches to principal component analysis in the presence of missing values.* The Journal of Machine Learning Research 11 (2010): 1957-2000.
-  Koren, Yehuda. *Factorization meets the neighborhood: a multifaceted collaborative filtering model.* Proceedings of the 14th ACM SIGKDD international conference on Knowledge discovery and data mining. ACM, 2008.