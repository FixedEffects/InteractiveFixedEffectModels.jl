
This package estimates factor models on datasets where each row represents an observation,  as opposed to a matrix N x T.


I'll use the term "panels" to refer to these long datasets , and id x time to refer to the two dimensions of the factor structure - they correspond to (variable x observation) in PCA and (user x movie) in recommandation problems.



Factor models are estimated by minimizing a sum of squares. This yields three importants benefits compared to an eigenvalue decomposition:

1. estimate unbalanced panels, i.e. with missing (id x time) observations. 

2. estimate weighted factor models, where weights are not constant within id or time

3. estimate factor models with a penalization for the norm of loadings and factors (Tikhonov regularization). 

   ```
   sum of errors + lambda *(||loadings||^2 + ||factors||^2)
   ```

4. avoid the creation of a matrix N x T, which may use a lot of memory

An alternative for issue 1 is the the EM algorithm, which replaces iteratively missing values by the predicted values from the factor model until convergence. However, the EM algorithm is generally slower to converge.



### Syntax
- The first argument of `fit` is an object of type `PanelFactorModel`. This must be constructed by specifying the id variable, the time variable, and the factor dimension in the dataframe. Both the id and time variable must be of type `PooledDataVector`.

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
	- WHen it is a symbol,` fit` fits a factor model on a variable. 

		```julia
		fit(PanelFactorModel(:pState, :pYear, 2), :Sales)
		```

	- When it is a formula, `fit` fits a linear model with interactive fixed effects (as in Bai (2000))
		Note that the id or time fixed effects can be specified using `|>` as in the [FixedEffectModels.jl](https://github.com/matthieugomez/FixedEffectModels.jl) package

		```julia
		fit(pfm, Sales ~ Price, df)
		fit(PanelFactorModel(:pState, :pYear, 2), Sales ~ Price |> pState + pYear, df)
		```


- `fit` also has keyword arguments:
	- `subset`
	- `weights`: This minimizes the sum of weighted residuals
	- `lambda` This option implements a Tikhonov regularization, i.e. minimizing the sum of residuals +  lambda( ||factors||^2 + ||loadings||^2)
	- `method`. This option allows to vary the optimization method used. It defaults to `:gradient_descent` when estimating a factor model, and `:bfgs` when estimating a linear model with interactive fixed effects.  You can also choose the method described in Bai (2009) by using `method = :svd`.


	## Install

	```julia
	Pkg.clone("https://github.com/matthieugomez/PanelFactorModels.jl")
	```