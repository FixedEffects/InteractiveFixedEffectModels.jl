[![Build Status](https://travis-ci.org/matthieugomez/SparseFactorModels.jl.svg?branch=master)](https://travis-ci.org/matthieugomez/SparseFactorModels.jl)
[![Coverage Status](https://coveralls.io/repos/matthieugomez/SparseFactorModels.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/matthieugomez/SparseFactorModels.jl?branch=master)

## Motivation

This package estimates linear factor models on sparse datasets (i.e. datasets where there is not one observation per (id x time) combination).

For an observation `i`, denote `(jλ(i), jf(i))` the associated pair (id x time).  This package estimates the set of coefficients `beta`, of factors `(f1, .., fr)` and of loadings `(λ1, ..., λr)` that solve

![minimization](img/minimization.png)


When X is a set of id or time dummies, this problem corresponds to a principal component analysis with missing values. When X is a general set of regressors, this problem corresponds to a linear model with interactive fixed effects as described in Bai (2009).


To install
```julia
Pkg.update()
Pkg.clone("https://github.com/matthieugomez/SparseFactorModels.jl")
```


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
		fit(SparseFactorModel(:pState, :pYear, 2), Sales ~ 0, df)
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

- `:ar` This method corresponds to coordinate gradient descent (= Gauss Seidel). 
- `:lm` This method corresponds to Levenberg Marquardt Method adapted for sparse matrices.


Note that I don't use the svd method described in Bai (2009) since it requires (i) that the initial dataset contains unique observations for a given pair id x time (ii) that there is enough RAM to store a matrix NxT.




#### weights

The `weights` option allows to minimize the sum of *weighted* residuals. This option is not available for the option `:svd`. 

When weights are not constant within id or time, the optimization problem has local minima that are not global - all methods may end up on such local minimum. You should therefore use weights that are constant within id or time or expereriment with different `methods`


#### errors
Compute robust standard errors by constructing an object of type `AbstractVcovMethod`. For now, `VcovSimple()` (default), `VcovWhite()` and `VcovCluster(cols)` are implemented.

```julia
fit(SparseFactorModel(:pState, :pYear, 2), Sales ~ Price, df, VcovCluster(:pState))
```

#### save
The option `save = true` saves a new dataframe storing residuals, factors, loadings and the eventual fixed effects. Importantly, the new dataframe is aligned with the initial dataframe (rows not used in the estimation are simply filled with NA).

## FAQ
#### When should I use interactive fixed effects?
Time fixed effects allow to control for aggregate shocks that impact individuals in the same way. Interactive fixed effects allow to control for aggregate shocks that impact individuals in different ways, as long as this heterogeneity is constant accross time.

You can find such models in the following articles:

- Eberhardt, Helmers, Strauss (2013) *Do spillovers matter when estimating private returns to R&D?*
- Hagedorn, Karahan, Movskii (2015) *Unemployment Benefits and Unemployment in the Great Recession: The Role of Macro Effects*
- Hagedorn, Karahan, Movskii (2015) *The impact of unemployment benefit extensions on employment: the 2014 employment miracle?* 
- Totty (2015) *The Effect of Minimum Wages on Employment: A Factor Model Approach*

#### How are standard errors computed?
The `cov` option is passed to a regression of y on x and covariates of the form `i.id#c.year` and `i.year#c.id`. This way of computing standard errors is hinted in section 6 of of Bai (2009).


#### What if I don't know the number of factors?
 Moon Weidner (2015) show that overestimating the number of factors still returns consistent estimates: irrelevant factors behave similarly to irrelevant covariates in a traditional OLS. A rule of thumb is to check that your estimate stays constant when you add more factors.

#### Does this command implement the bias correction term in Bai (2009)?
In presence of cross or time correlation beyond the factor structure, the estimate for beta is biased (but still consistent): see Theorem 3 in Bai 2009, which derives the correction term in special cases. However, this package does not implement any correction. You may want to add enough factors until residuals are approximately i.i.d.


#### Can `β` be estimated by replacing X with the residuals of X on a factor model?
For models with fixed effect, an equivalent way to obtain β is to first demean regressors within groups and then regress `y` on these residuals instead of the original regressors.
In contrast, this method does not work with models with interactive fixed effects. While fixed effects are linear projections (so that the Frisch-Waugh-Lovell theorem holds), factor models are non linear projections.

#### Can I have multiple observations per (id x time) ?
Yes, unless you use the method `svd`. That being said, be aware of local minima by checking different methods give the same results.





## References
- Bai, Jushan. *Panel data models with interactive fixed effects.* (2009) Econometrica 
- Ilin, Alexander, and Tapani Raiko. *Practical approaches to principal component analysis in the presence of missing values.* (2010) The Journal of Machine Learning Research 11 
-  Koren, Yehuda. *Factorization meets the neighborhood: a multifaceted collaborative filtering model.* (2008) Proceedings of the 14th ACM SIGKDD international conference on Knowledge discovery and data mining. 
- Raiko, Tapani, Alexander Ilin, and Juha Karhunen. *Principal component analysis for sparse high-dimensional data.* (2008) Neural Information Processing.
- Srebro, Nathan, and Tommi Jaakkola. *Weighted low-rank approximations* (2010) The Journal of Machine Learning Research 11 
- Nocedal, Jorge and Stephen Wright *An Inexact Levenberg-Marquardt method for Large Sparse Nonlinear Least Squares*  (1985) The Journal of the Australian Mathematical Society

## Related Packages
- https://github.com/joidegn/FactorModels.jl : fits and predict factor models on matrices
- https://github.com/madeleineudell/LowRankModels.jl : fits general low rank approximations on matrices
- https://github.com/aaw/IncrementalSVD.jl: implementation of the backpropagation algorithm

