[![Build status](https://github.com/FixedEffects/InteractiveFixedEffectModels.jl/workflows/CI/badge.svg)](https://github.com/FixedEffects/InteractiveFixedEffectModels.jl/actions)

# InteractiveFixedEffectModels.jl

`InteractiveFixedEffectModels.jl` estimates interactive fixed effect models of the Bai (2009) form:

![minimization](img/minimization.png)

The package is designed for panel settings with latent factors, optional high-dimensional fixed effects on the same panel dimensions, weights, clustering, and incomplete panels.

It integrates with the Julia `StatsModels` / `FixedEffectModels` ecosystem:

- formulas use `@formula(...)`
- high-dimensional fixed effects use `fe(...)`
- covariance estimators come from `Vcov.jl`

## Installation

The package is registered in the General registry:

```julia
] add InteractiveFixedEffectModels
```

Current package compatibility requires Julia `1.9+`.

## Quick Start

```julia
using DataFrames, RDatasets, InteractiveFixedEffectModels

df = dataset("plm", "Cigar")

result = regife(
    df,
    @formula(Sales ~ Price + ife(State, Year, 2) + fe(State))
)

display(result)
```

Typical output:

```text
Interactive Fixed Effect Model
Number of obs: ...
R²: ...
R² within: ...
```

## Model Specification

Every `regife` formula must contain exactly one interactive fixed effect term:

```julia
ife(id, time, rank)
```

where:

- `id` is the unit dimension
- `time` is the time dimension
- `rank` is the number of latent factors

Example:

```julia
@formula(Sales ~ Price + ife(State, Year, 2))
```

High-dimensional fixed effects can also be added with `fe(...)`, but only on the same panel dimensions used in `ife(...)`:

```julia
@formula(Sales ~ Price + ife(State, Year, 2) + fe(State))
@formula(Sales ~ Price + ife(State, Year, 2) + fe(Year))
@formula(Sales ~ Price + ife(State, Year, 2) + fe(State) + fe(Year))
```

This is invalid:

```julia
@formula(Sales ~ Price + ife(State, Year, 2) + fe(Region))
```

unless `Region` is the same dimension as one of the `ife(...)` variables.

## Main Options

### Covariance Estimators

Pass a `Vcov.jl` covariance estimator as the third positional argument:

```julia
regife(df, @formula(y ~ x + ife(id, time, 2)), Vcov.simple())
regife(df, @formula(y ~ x + ife(id, time, 2)), Vcov.robust())
regife(df, @formula(y ~ x + ife(id, time, 2)), Vcov.cluster(:id))
regife(df, @formula(y ~ x + ife(id, time, 2)), Vcov.cluster(:id, :time))
```

### Weights

Use a positive weight column:

```julia
regife(df, @formula(y ~ x + ife(id, time, 2)); weights = :w)
```

### Subsetting

Restrict estimation to a subset while keeping saved outputs aligned with the original data:

```julia
regife(df, @formula(y ~ x + ife(id, time, 2)); subset = df.year .>= 1980)
```

### Optimization Method

Available methods are:

- `:dogleg`
- `:levenberg_marquardt`
- `:gauss_seidel`

Example:

```julia
regife(df, @formula(y ~ x + ife(id, time, 2)); method = :dogleg)
```

### Saved Outputs

With `save = true`, the returned object contains an `augmentdf` aligned with the original data. Depending on the specification, it can include:

- `residuals`
- `factors1`, `factors2`, ...
- `loadings1`, `loadings2`, ...
- absorbed fixed effects such as `fe_State`

Rows excluded from estimation are filled with `missing`.

## Worked Examples

### Interactive Fixed Effect Regression

```julia
using DataFrames, RDatasets, InteractiveFixedEffectModels

df = dataset("plm", "Cigar")

result = regife(
    df,
    @formula(Sales ~ Price + ife(State, Year, 2) + fe(State)),
    Vcov.cluster(:State);
    save = true
)

coef(result)
result.augmentdf
```

### Factor Model / PCA-Like Use

Factor models are a special case with no observed regressors.

```julia
using DataFrames, RDatasets, InteractiveFixedEffectModels

df = dataset("plm", "Cigar")

result = regife(
    df,
    @formula(Sales ~ 0 + ife(State, Year, 2));
    save = true
)
```

To demean with respect to one panel dimension:

```julia
regife(df, @formula(Sales ~ ife(State, Year, 2) + fe(State)); save = true)
```

### Programmatic Formula Construction

```julia
using StatsModels

regife(
    df,
    Term(:Sales) ~ Term(:Price) + ife(Term(:State), Term(:Year), 2) + fe(Term(:State))
)
```

## Returned Objects

If the model includes regressors, `regife` returns an `InteractiveFixedEffectModel`, which behaves like a regression result and supports methods such as:

- `coef`
- `coefnames`
- `vcov`
- `confint`
- `coeftable`
- `nobs`
- `r2`
- `adjr2`

If the model contains no regressors, it returns a lighter `FactorResult` with estimation metadata and the optional saved dataframe.

## Notes and Caveats

### Local Minima

The package supports:

- missing observations within the `id × time` panel
- multiple observations per `id × time`
- weights

In those cases, the optimization problem can have local minima. The package includes a restart heuristic to reduce that risk, but it is still a numerical optimization problem rather than a closed-form estimator.

### Standard Errors

The reported standard errors are based on regressing on covariates of the form:

- `i.id # c.time`
- `i.time # c.id`

This follows the discussion in Section 6 of Bai (2009).

### Bias Correction

The package does not implement the Bai (2009) bias correction for settings with residual cross-sectional or serial dependence beyond the factor structure. In such cases, coefficient estimates can remain consistent but biased in finite samples.

### When should one use interactive fixed effects models?

Some literature using this estimation procedure:

- Eberhardt, Helmers, Strauss (2013), *Do spillovers matter when estimating private returns to R&D?*
- Hagedorn, Karahan, Manovskii (2015), *Unemployment Benefits and Unemployment in the Great Recession: The Role of Macro Effects*
- Hagedorn, Karahan, Manovskii (2015), *The impact of unemployment benefit extensions on employment: the 2014 employment miracle?*
- Totty (2015), *The Effect of Minimum Wages on Employment: A Factor Model Approach*

## Related Packages

- [FixedEffectModels.jl](https://github.com/matthieugomez/FixedEffectModels.jl): linear models with high-dimensional fixed effects
- [FactorModels.jl](https://github.com/joidegn/FactorModels.jl): factor models on matrices
- [LowRankModels.jl](https://github.com/madeleineudell/LowRankModels.jl): general low-rank approximation models

## Reference

- Bai, Jushan (2009), "Panel Data Models With Interactive Fixed Effects," *Econometrica*.
