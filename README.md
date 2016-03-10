[![Build Status](https://travis-ci.org/matthieugomez/SparseFactorModels.jl.svg?branch=master)](https://travis-ci.org/matthieugomez/SparseFactorModels.jl)
[![Coverage Status](https://coveralls.io/repos/matthieugomez/SparseFactorModels.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/matthieugomez/SparseFactorModels.jl?branch=master)

## Motivation



This problem estimates factor models, generalized along two dimensions:

-  Why a usual PCA requires one and only one observation per combination id x time, this package estimates models with multiple observations by combination id x time (for instance group level factors) or missing combinations (as in the Netflix problem, with ratings by user x movies).

- Beyond factors and loadings, this package allows models with linear regressors . This corresponds to the Bai (2009) linear model with interactive fixed effect. The interaction between factors and loadings allows to control for shocks with heterogeneous impacts accross ids, as long as this heterogeneity is constant accross some other dimension (like time).

Formally, denote `(id(i), time(i))` the combination associated to an observation `i`.  This package estimates the set of coefficients `β`, of factors `(f1, .., fr)` and of loadings `(λ1, ..., λr)` that solve

![minimization](img/minimization.png)



Threee minimization methods are available:
- `:gauss_seidel` (corresponds to coordinate gradient descent)
- `:levenberg_marquardt`
- `:dogleg` 

To install

```julia
Pkg.clone("https://github.com/matthieugomez/SparseFactorModels.jl")
```

## Syntax

The general syntax is
```julia
fit(pfm::SparseFactorModel,
	f::Formula, 
    df::AbstractDataFrame, 
    vcov_method::AbstractVcovMethod = VcovSimple();
 	method::Symbol = :dogleg
    weight::Union{Symbol, Void} = nothing, 
    subset::Union{AbstractVector{Bool}, Void} = nothing,
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



#### errors
Compute robust standard errors by constructing an object of type `AbstractVcovMethod`. For now, `VcovSimple()` (default), `VcovWhite()` and `VcovCluster(cols)` are implemented.

```julia
fit(SparseFactorModel(:pState, :pYear, 2), Sales ~ Price, df, VcovCluster(:pState))
```

#### save
The option `save = true` saves a new dataframe storing residuals, factors, loadings and the eventual fixed effects. Importantly, the new dataframe is aligned with the initial dataframe (rows not used in the estimation are simply filled with NA).


#### weights and multiple observations

The package handles situations with weights that are not constant within id or time or/and multiple observations per id x time pair. However, in this case, the optimization problem tends to have local minima. The algorithm tries to catch these cases, and, when this happens, the optimization algorithm is restarted on a random starting point. However I'm not sure all cases are caught. 

## FAQ
#### When should one use interactive fixed effects models?
Some litterature using this estimation procedure::

- Eberhardt, Helmers, Strauss (2013) *Do spillovers matter when estimating private returns to R&D?*
- Hagedorn, Karahan, Movskii (2015) *Unemployment Benefits and Unemployment in the Great Recession: The Role of Macro Effects*
- Hagedorn, Karahan, Movskii (2015) *The impact of unemployment benefit extensions on employment: the 2014 employment miracle?* 
- Totty (2015) *The Effect of Minimum Wages on Employment: A Factor Model Approach*

#### How are standard errors computed?
Errors are obtained by regressing y on x and covariates of the form `i.id#c.year` and `i.year#c.id`. This way of computing standard errors is hinted in section 6 of of Bai (2009).


#### What if the number of factors is unkown?
 Moon Weidner (2015) show that overestimating the number of factors returns consistent estimates: irrelevant factors behave similarly to irrelevant covariates in a traditional OLS. 

#### Does this command implement the bias correction term in Bai (2009)?
In presence of cross or time correlation beyond the factor structure, the estimate for beta is consistent but biased (see Theorem 3 in Bai 2009, which derives the correction term in special cases). However, this package does not implement any correction. You may want to check that your residuals are approximately i.i.d.


#### Can't the model be estimated by replacing X with the residuals of X on a factor model?
No. This two-step method works for fixed effect models (Frisch-Waugh-Lovell theorem), but does not work for interactive fixed effect models. Factor models are not linear projections.




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

