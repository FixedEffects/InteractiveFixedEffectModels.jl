# Object constructed by the user
type PanelFactorModel
    id::Symbol
    time::Symbol
    rank::Int64
end

# object returned by fitting variable
abstract AbstractPanelFactorResult

type PanelFactorResult <: AbstractPanelFactorResult
    coef::Vector{Float64}

    esample::BitVector
    augmentdf::DataFrame

    iterations::Vector{Int64}
    converged::Vector{Bool}
end

##############################################################################
##
## Starts from a dataframe and returns a dataframe
##
##############################################################################

function fit(m::PanelFactorModel, f::Formula, df::AbstractDataFrame; method::Symbol = :l_bfgs, lambda::Real = 0.0, subset::Union(AbstractVector{Bool}, Nothing) = nothing, weight::Union(Symbol, Nothing) = nothing, maxiter::Integer = 10000, tol::Real = 1e-9, save = true)

	##############################################################################
	##
	## Prepare DataFrame
	##
	##############################################################################

	# initial check
	if method == :svd
		weight == nothing || error("The svd methods does not work with weights")
		lambda == 0.0 || error("The svd method only works with lambda = 0.0")
	elseif method == :gs
		lambda == 0.0 || error("The Gauss-Seidel method only works with lambda = 0.0")
	end


	rf = deepcopy(f)
	## decompose formula into normal  vs absorbpart
	(has_absorb, absorb_formula, absorb_terms, has_iv, iv_formula, iv_terms, endo_formula, endo_terms) = decompose!(rf)
	if has_iv
		error("partial_out does not support instrumental variables")
	end
	rt = Terms(rf)
	has_regressors = allvars(rf.rhs) != [] || rt.intercept == true

	## create a dataframe without missing values & negative weights
	vars = allvars(rf)
	absorb_vars = allvars(absorb_formula)
	factor_vars = [m.id, m.time]
	all_vars = vcat(vars, absorb_vars, factor_vars)
	all_vars = unique(convert(Vector{Symbol}, all_vars))
	esample = complete_cases(df[all_vars])
	if weight != nothing
		esample &= isnaorneg(df[weight])
		all_vars = unique(vcat(all_vars, weight))
	end
	if subset != nothing
		length(subset) == size(df, 1) || error("the number of rows in df is $(size(df, 1)) but the length of subset is $(size(df, 2))")
		esample &= convert(BitArray, subset)
	end
	subdf = df[esample, all_vars]
	all_except_absorb_vars = unique(convert(Vector{Symbol}, vcat(vars, factor_vars)))
	for v in all_except_absorb_vars
		dropUnusedLevels!(subdf[v])
	end

	# Compute weight
	sqrtw = get_weight(subdf, weight)

	## Compute factors, an array of AbtractFixedEffects
	if has_absorb
		fes = AbstractFixedEffect[FixedEffect(subdf, a, sqrtw) for a in absorb_terms.terms]
		# in case some FixedEffect is aFixedEffectIntercept, remove the intercept
		if any([typeof(f) <: FixedEffectIntercept for f in fes]) 
			rt.intercept = false
		end
	else
		fes = nothing
	end

	# initialize iterations and converged
	iterations = Int[]
	converged = Bool[]

	# get two dimensions
	id = subdf[m.id]
	time = subdf[m.time]

	## Compute demeaned X
	mf = simpleModelFrame(subdf, rt, esample)
	if has_regressors
		coef_names = coefnames(mf)
		X = ModelMatrix(mf).m
		broadcast!(*, X, X, sqrtw)
		demean!(X, iterations, converged, fes)
	end

	## Compute demeaned y
	py = model_response(mf)[:]
	if eltype(py) != Float64
		y = convert(py, Float64)
	else
		y = py
	end
	broadcast!(*, y, y, sqrtw)
	demean!(y, iterations, converged, fes)

	##############################################################################
	##
	## Estimate Model
	##
	##############################################################################

	## Case regressors
	if has_regressors
		# initial b
		crossx = cholfact!(At_mul_B(X, X))
		coef =  crossx \ At_mul_B(X, y)

		# dispatch manually to the right method
		if method == :svd
			M = crossx \ X'
			(coef, loadings, factors, iterations, converged) = fit_svd(X, M, coef, y, id, time, m.rank, maxiter = maxiter, tol = tol) 
		elseif method == :gs
			M = crossx \ X'
			(coef, loadings, factors, iterations, converged) = fit_gs(X, M, coef, y, id, time, m.rank, sqrtw, maxiter = maxiter, tol = tol) 
		else
			(coef, loadings, factors, iterations, converged) = fit_optimization(X, coef, y, id, time, m.rank,  sqrtw, method = method, lambda = lambda, maxiter = maxiter, tol = tol) 
		end
	else
		coef = [0.0]
		if method == :svd
			(loadings, factors, iterations, converged) = fit_svd(y, id, time, m.rank, maxiter = maxiter, tol = tol)
		elseif method == :gs
			(loadings, factors, iterations, converged) = fit_gs(y, id, time, m.rank, sqrtw, maxiter = maxiter, tol = tol)
		elseif method == :backpropagation
		    (loadings, factors, iterations, converged) = fit_backpropagation(y, id, time, m.rank, sqrtw, regularizer = 0.001, learning_rate = 0.001, maxiter = maxiter, tol = tol)
		else
		    (loadings, factors, iterations, converged) = fit_optimization(y, id, time, m.rank, sqrtw, lambda = lambda, method = method, maxiter = maxiter, tol = tol)
		end
	end

	##############################################################################
	##
	## Return result
	##
	##############################################################################


	if save 
		# factors and loadings
		augmentdf = DataFrame(id, time, loadings, factors, esample)

		# residuals
		res = deepcopy(y)
		if has_regressors
			subtract_b!(res, y, coef, X)
		end
		for r in 1:m.rank
			subtract_factor!(res, sqrtw, id.refs, loadings, time.refs, factors, r)
		end
		broadcast!(/, res, res, sqrtw)
		if all(esample)
			augmentdf[:residuals] = res
		else
			augmentdf[:residuals] =  DataArray(Float64, size(augmentdf, 1))
			augmentdf[esample, :residuals] = res
		end


		# fixed effects
		if has_absorb
			# residual before demeaning
			mf = simpleModelFrame(subdf, rt, esample)
			oldy = model_response(mf)[:]
			if has_regressors
				oldres = similar(oldy)
				oldX = ModelMatrix(mf).m
				subtract_b!(oldres, oldy, coef, oldX)
			else
				oldres = oldy
			end
			for r in 1:m.rank
				subtract_factor!(oldres, fill(one(Float64), length(res)), id.refs, loadings, time.refs, factors, r)
			end
			b = oldres - res
			# get fixed effect
			augmentdf = hcat(augmentdf, getfe(fes, b, esample))
		end
	else
		augmentdf = DataFrame()
	end

	PanelFactorResult(coef, esample, augmentdf, iterations, converged)
end


# Symbol to formul Symbol ~ 0
function fit(m::PanelFactorModel, variable::Symbol, df::AbstractDataFrame; method::Symbol = :gradient_descent, lambda::Real = 0.0, subset::Union(AbstractVector{Bool}, Nothing) = nothing, weight::Union(Symbol, Nothing) = nothing, maxiter::Integer = 10000, tol::Real = 1e-8)
    formula = Formula(variable, 0)
    fit(m, formula, df, method = method, lambda = lambda, subset = subset, weight = weight, subset = subset, maxiter = maxiter, tol = tol, save = true)
end

##############################################################################
##
## DataFrame from factors loadings
##
##############################################################################

function DataFrame(id::PooledDataVector, time::PooledDataVector, loadings::Matrix{Float64}, factors::Matrix{Float64}, esample::BitVector)
	df = DataFrame()
	anyNA = all(esample)
	for r in 1:size(loadings, 2)
		# loadings
		df[convert(Symbol, "loadings$r")] = build_column(id, loadings, r, esample)
		df[convert(Symbol, "factors$r")] = build_column(time, factors, r, esample)
	end
	return df
end


function build_column(id::PooledDataVector, loadings::Matrix{Float64}, r::Int, esample::BitVector)
	T = eltype(id.refs)
	refs = fill(zero(T), length(esample))
	refs[esample] = id.refs
	return PooledDataArray(RefArray(refs), loadings[:, r])
end
