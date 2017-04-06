[![Build Status](https://travis-ci.org/matthieugomez/InteractiveFixedEffectModels.jl.svg?branch=master)](https://travis-ci.org/matthieugomez/InteractiveFixedEffectModels.jl)
[![Coverage Status](https://coveralls.io/repos/matthieugomez/InteractiveFixedEffectModels.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/matthieugomez/InteractiveFixedEffectModels.jl?branch=master)

## Install

```julia
Pkg.add("InteractiveFixedEffectModels")
```


## Motivation
This package implements a novel, fast and robust algorithm to estimate interactive fixed effect models (Bai 2009).


Formally, denote `T(i)` and `I(i))` the two categorical dimensions associated with observation `i` (typically time and id).  This package estimates the set of coefficients `β`, of factors `(f1, .., fr)` and of loadings `(λ1, ..., λr)` in the model

![minimization](img/minimization.png)




#### formula
A typical formula is composed of one dependent variable and regressors
```
using RDatasets, DataFrames, InteractiveFixedEffectModels
df = dataset("plm", "Cigar")
```
When the only regressor is `0`, `fit` fits a factor model on the left hand side variable
```julia
@formula(Sales ~ 0)
```
With multiple regressors, `fit` fits a linear model with interactive fixed effects (Bai (2009))
```julia
@formula(Sales ~ Price)
```
### `ife`
Interactive fixed effects are indicated with the macro `@ife`. The id and time variables must refer to variables of type `PooledDataVector`.

```julia
df[:pState] =  pool(df[:State])
df[:pYear] =  pool(df[:Year])
@ife(pState + pYear, 2)
```

### `fe`
Fixed effects are indicated with the macro `@fe`. Use only the variables specified in the factor model. See [FixedEffectModels.jl](https://github.com/matthieugomez/FixedEffectModels.jl) for more information

```julia
@fe(pState)
@fe(pYear)
@fe(pState + pYear)
```

### standard errors 
Standard errors are indicated with the macro `@vcovrobust()` or `@vcovcluster()`
```julia
@vcovrobust()
@vcovcluster(StatePooled)
@vcovcluster(StatePooled, YearPooled)
```

#### weight
weights are indicated with the macro `@weight`
```julia
@weight(Pop)
```


### options

- Two minimization methods are available:
	- `:levenberg_marquardt`
	- `:dogleg` 

- The option `save = true` saves a new dataframe storing residuals, factors, loadings and the eventual fixed effects. Importantly, the returned dataframe is aligned with the initial dataframe (rows not used in the estimation are simply filled with NA).

###  Putting everything together
```julia
using DataFrames, RDatasets, FixedEffectModels
df = dataset("plm", "Cigar")
reg(df, @formula(Sales ~ Price), @ife(pState + pYear, 2), @fe(pState), method = method, save = true)
#=====================================================================
# Number of obs                1380   Degree of freedom              93
# R2                          0.245   R2 Adjusted                 0.190
# F Stat                    417.342   p-val                       0.000
# Iterations                      2   Converged:                   true
# =====================================================================
#         Estimate   Std.Error t value Pr(>|t|)   Lower 95%   Upper 95%
# ---------------------------------------------------------------------
# NDI  -0.00568607 0.000278334 -20.429    0.000 -0.00623211 -0.00514003
# =====================================================================
```


## Unicity
The algorithm can estimate models with missing observations per id x time, multiple observations per id x time, and weights (see below).

With multiple observations per id x time, or with weights non constant within id or time, the optimization problem may have local minima. The algorithm tries to catch these cases, and, when this happens, the optimization algorithm is restarted on a random starting point. However I'm not sure all cases are caught. 

## FAQ
#### When should one use interactive fixed effects models?
Some litterature using this estimation procedure::

- Eberhardt, Helmers, Strauss (2013) *Do spillovers matter when estimating private returns to R&D?*
- Hagedorn, Karahan, Movskii (2015) *Unemployment Benefits and Unemployment in the Great Recession: The Role of Macro Effects*
- Hagedorn, Karahan, Movskii (2015) *The impact of unemployment benefit extensions on employment: the 2014 employment miracle?* 
- Totty (2015) *The Effect of Minimum Wages on Employment: A Factor Model Approach*

#### How are standard errors computed?
Errors are obtained by regressing y on x and covariates of the form `i.id#c.year` and `i.year#c.id`. This way of computing standard errors is hinted in section 6 of of Bai (2009).


#### Does this command implement the bias correction term in Bai (2009)?
In presence of cross or time correlation beyond the factor structure, the estimate for beta is consistent but biased (see Theorem 3 in Bai 2009, which derives the correction term in special cases). However, this package does not implement any correction. You may want to check that your residuals are approximately i.i.d.


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

