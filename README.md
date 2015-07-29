
## Motivation

This package estimates linear factor models on sparse datasets (as in the Netflix problem).

For an observation `i`, denote `(jλ(i), jf(i))` the associated pair (id x time).  This package estimates the set of coefficients `beta`, of factors `(f1, .., fr)` and of loadings `(λ1, ..., λr)` that solve

![minimization](img/minimization.png)


When X is a set of id or time dummies, this problem corresponds to a principal component analysis with missing values. When X is a general set of regressors, this problem corresponds to a linear model with interactive fixed effects as described in Bai (2009).


## Syntax

The general syntax is
```julia
fit(pfm::SparseFactorModel,
	f::Formula, 
    df::AbstractDataFrame, 
    vcov_method::AbstractVcovMethod = VcovSimple();
 	method::Symbol = :gs
    weight::Union(Symbol, Nothing) = nothing, 
    subset::Union(AbstractVector{Bool}, Nothing) = nothing,
    save::Bool = true, 
    maxiter::Int64 = 10000, tol::Float64 = 1e-8
    )
```


- The first argument of `fit` is an object of type `SparseFactorModel`. Such an object can be constructed by specifying the id variable, the time variable, and the rank of the factor model. Both the id and time variable must be of type `PooledDataVector`.

	```julia
	using RDatasets, DataFrames, SparseFactorModels
	df = dataset("plm", "Cigar")
	# create PooledDataVector
	df[:pState] =  pool(df[:State])
	df[:pYear] =  pool(df[:Year])
	# create SparseFactorModel in state, year, and rank 2
	factormodel = SparseFactorModel(:pState, :pYear, 2)
	```

- The second argument of `fit` is a formula
	- When the only regressor is `0`, `fit` fits a factor model on the left hand side variable

		```julia
		fit(SparseFactorModel(:pState, :pYear, 2), Sales ~ 0)
		```

		You can pre-demean the variable using `|>` as in the package [FixedEffectModels.jl](https://github.com/matthieugomez/FixedEffectModels.jl).

		```julia
		fit(SparseFactorModel(:pState, :pYear, 2), Sales ~ 1 |> pState, df)
		fit(SparseFactorModel(:pState, :pYear, 2), Sales ~ 1 |> pYear, df)
		fit(SparseFactorModel(:pState, :pYear, 2), Sales ~ 1 |> pState + pYear, df)
		```

	- With multiple regressors, `fit` fits a linear model with interactive fixed effects (Bai (2009))
	

		```julia
		fit(SparseFactorModel(:pState, :pYear, 2), Sales ~ Price, df)
		```

		Similarly, you may add id  or time fixed effects
		```julia
		fit(SparseFactorModel(:pState, :pYear, 2), Sales ~ Price |> pState, df)
		```


#### method
Three methods are available

- `:gs` (default for interactive fixed effect models) This method fits a factor model by alternating regressions on loadings interacted with time dummy and factors interacted by id dummy, as in the [Gauss-Seidel method](https://en.wikipedia.org/wiki/Gauss%E2%80%93Seidel_method). 


- `:svd`. This method fits a factor model by SVD on the matrix N x T, after imputing missing values using the factor model fitted in the previous iteration. The `:svd` method requires that the initial dataset contains unique observations for a given pair id x time, and that there is enough RAM to store a matrix NxT. The `svd` method is fast when T/N is small and when the number of missing values is small.


- Optimization methods (such as `:gradient_descent`, `:momentum_gradient_descent`). These methods estimate the model by directly minimizing the sum of squared residuals using the Package Optim. 


The default for factor models is `:momentum_gradient_descent` and the default for interactive fixed effects is `:gs`.
You may find some speed comparisons [here](benchmark/benchmark.md)



#### weights

The `weights` option allows to minimize the sum of *weighted* residuals. This option is not available for the option `:svd`. 

When weights are not constant within id or time, the optimization problem has local minima that are not global. You may want to use the method `:gs` rather than an optimization method


#### errors
Compute robust standard errors by constructing an object of type `AbstractVcovMethod`. For now, `VcovSimple()` (default), `VcovWhite()` and `VcovCluster(cols)` are implemented.

```julia
fit(SparseFactorModel(:pState, :pYear, 2), Sales ~ Price, df, VcovCluster(:pState))
```

#### lambda
`lambda` adds a Tikhonov regularization term to the sum of squared residuals. This option is only available when using an optimization method.

#### save
The option `save = true` saves a new dataframe storing residuals, factors, loadings and the eventual fixed effects. Importantly, the new dataframe is aligned with the initial dataframe (rows not used in the estimation are simply filled with NA).

## Install

```julia
Pkg.clone("https://github.com/matthieugomez/SparseFactorModels.jl")
```

## References
- Bai, Jushan. *Panel data models with interactive fixed effects.* (2009) Econometrica 
- Ilin, Alexander, and Tapani Raiko. *Practical approaches to principal component analysis in the presence of missing values.* (2010) The Journal of Machine Learning Research 11 
-  Koren, Yehuda. *Factorization meets the neighborhood: a multifaceted collaborative filtering model.* (2008) Proceedings of the 14th ACM SIGKDD international conference on Knowledge discovery and data mining. 
- Raiko, Tapani, Alexander Ilin, and Juha Karhunen. *Principal component analysis for sparse high-dimensional data.* (2008) Neural Information Processing.
- Srebro, Nathan, and Tommi Jaakkola. *Weighted low-rank approximations* (2010) The Journal of Machine Learning Research 11 

## Related Packages
- https://github.com/joidegn/FactorModels.jl : fits and predict factor models on matrices
- https://github.com/madeleineudell/LowRankModels.jl : fits general low rank approximations on matrices
- https://github.com/aaw/IncrementalSVD.jl: implementation of the backpropagation algorithm

