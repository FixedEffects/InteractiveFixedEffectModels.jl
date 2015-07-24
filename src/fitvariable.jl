##############################################################################
##
## From DataFrame
##
##############################################################################

function fit(m::PanelFactorModel, variable::Symbol, df::AbstractDataFrame; method::Symbol = :gradient_descent, lambda::Real = 0.0, subset::Union(AbstractVector{Bool}, Nothing) = nothing, weight::Union(Symbol, Nothing) = nothing, maxiter::Integer = 10000, tol::Real = 1e-8)

	# initial check
	if method == :svd
		weight == nothing || error("The svd methods does not work with weights")
		lambda == 0.0 || error("The svd method only works with lambda = 0.0")
	elseif method == :reg
		lambda == 0.0 || error("The reg method only works with lambda = 0.0")
	end

    # Transform symbols + dataframe into real vectors
    factor_vars = [m.id, m.time]
    all_vars = vcat(variable, factor_vars)
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

    if weight == nothing
        w = fill(one(Float64), size(subdf, 1))
        sqrtw = w
    else
        w = convert(Vector{Float64}, subdf[weight])
        sqrtw = sqrt(w)
    end

    if typeof(subdf[variable]) == Vector{Float64}
        y = deepcopy(df[variable])
    else
        y = convert(Vector{Float64}, subdf[variable])
    end
    broadcast!(*, y, y, sqrtw)

    id = subdf[m.id]
    time = subdf[m.time]

    if method == :svd
    	fit_svd(y, id, time, m.rank, maxiter = maxiter, tol = tol)
    elseif method == :backpropagation
        fit_backpropagation(y, id, time, m.rank, sqrtw, regularizer = 0.001, learning_rate = 0.001, maxiter = maxiter, tol = tol)
    else
        fit_optimization(y, id, time, m.rank, sqrtw, lambda = lambda, method = method, maxiter = maxiter, tol = tol)
    end
end



##############################################################################
##
## Estimate factor model by incremental optimization routine
##
##############################################################################

function fit_optimization{Tid, Rid, Ttime, Rtime}(y::Vector{Float64}, id::PooledDataVector{Tid, Rid}, time::PooledDataVector{Ttime, Rtime}, rank::Integer, sqrtw::Vector{Float64}; lambda::Real = 0.0,  method::Symbol = :gradient_descent, maxiter::Integer = 10000, tol::Real = 1e-9)
    
    # initialize
    iterations = fill(maxiter, rank)
    converged = fill(false, rank)
    loadings = Array(Float64, (length(id.pool), rank))
    factors = Array(Float64, (length(time.pool), rank))

    # squeeze (loadings and factors) -> x0
    l = length(id.pool)
    x0 = fill(0.1, length(id.pool) + length(time.pool))
    for r in 1:rank
        # set up optimization problem
        f = x -> sum_of_squares(x, sqrtw, y, time.refs, id.refs, l, lambda)
        g! = (x, storage) -> sum_of_squares_gradient!(x, storage, sqrtw, y, time.refs, id.refs, l, lambda)
        fg! = (x, storage) -> sum_of_squares_and_gradient!(x, storage, sqrtw, y, time.refs, id.refs, l, lambda)
        d = DifferentiableFunction(f, g!, fg!)

        # optimize
        # xtol corresponds to maxdiff(x, x_previous)
        result = optimize(d, x0, method = method, iterations = maxiter, xtol = tol, ftol = 1e-32, grtol = 1e-32)
        
        # develop minimumm -> (loadings and factors)
        loadings[:, r] = result.minimum[1:l]
        factors[:, r] = result.minimum[(l+1):end]
        iterations[r] = result.iterations
        converged[r] = result.x_converged || result.f_converged || result.gr_converged
        
        # take the residuals y - lambda_i * ft
        subtract_factor!(y, sqrtw, id.refs, loadings, time.refs, factors, r)
    end
    (newloadings, newfactors) = rescale(loadings, factors)

    PanelFactorResult(id, time, newloadings, newfactors, iterations, converged)
end

# fitness
function sum_of_squares{Ttime, Tid}(x::Vector{Float64}, sqrtw::Vector{Float64}, y::Vector{Float64}, timerefs::Vector{Ttime}, idrefs::Vector{Tid}, l::Integer, lambda::Real)
    out = zero(Float64)
    @inbounds @simd for i in 1:length(y)
        idi = idrefs[i]
        timei = timerefs[i] + l
        loading = x[idi]
        factor = x[timei]
        sqrtwi = sqrtw[i]
        error = y[i] - sqrtwi * loading * factor
        out += abs2(error)
    end
    
    # Tikhonov term
    @inbounds @simd for i in 1:length(x)
        out += lambda * abs2(x[i])
    end
    return out
end

# gradient
function sum_of_squares_gradient!{Ttime, Tid}(x::Vector{Float64}, storage::Vector{Float64}, sqrtw::Vector{Float64}, y::Vector{Float64}, timerefs::Vector{Ttime}, idrefs::Vector{Tid}, l::Integer, lambda::Real)
    fill!(storage, zero(Float64))
    @inbounds @simd for i in 1:length(y)
        idi = idrefs[i]
        timei = timerefs[i] + l
        loading = x[idi]
        factor = x[timei]
        sqrtwi = sqrtw[i]
        error = y[i] - sqrtwi * loading * factor
        storage[idi] -= 2.0 * error * sqrtwi * factor 
        storage[timei] -= 2.0 * error * sqrtwi * loading
    end
    
    # Tikhonov term
    @inbounds @simd for i in 1:length(x)
        storage[i] += 2.0 * lambda * x[i]
    end
    return storage
end

# fitness + gradient in the same loop
function sum_of_squares_and_gradient!{Ttime, Tid}(x::Vector{Float64}, storage::Vector{Float64}, sqrtw::Vector{Float64}, y::Vector{Float64}, timerefs::Vector{Ttime}, idrefs::Vector{Tid}, l::Integer, lambda::Real)
    fill!(storage, zero(Float64))
    len_y = length(y)
    out = zero(Float64)
    @inbounds @simd for i in 1:length(y)
        idi = idrefs[i]
        timei = timerefs[i]+l
        loading = x[idi]
        factor = x[timei]
        sqrtwi = sqrtw[i]
        error =  y[i] - sqrtwi * loading * factor
        out += abs2(error)
        storage[idi] -= 2.0 * error * sqrtwi * factor 
        storage[timei] -= 2.0 * error * sqrtwi * loading 
    end
    
    # Tikhonov term
    @inbounds @simd for i in 1:length(x)
        out += lambda * abs2(x[i])
        storage[i] += 2.0 * lambda * x[i]
    end
    return out
end


##############################################################################
##
## Estimate factor model by EM Method
##
##############################################################################

function fit_svd{Tid, Rid, Ttime, Rtime}(y::Vector{Float64}, ids::PooledDataVector{Tid, Rid}, times::PooledDataVector{Ttime, Rtime}, rank::Integer; maxiter::Integer = 10000, tol::Real = 1e-8)
 
    # initialize at zero for missing values
    res_matrix = fill(zero(Float64), (length(ids.pool), length(times.pool)))
    predict_matrix = deepcopy(res_matrix)
    factors = Array(Float64, (length(times.pool), rank))
    variance = Array(Float64, (length(times.pool), length(times.pool)))
    converged = Bool[false]
    iterations = Int[maxiter]
    error = zero(Float64)
    olderror = zero(Float64)

    # starts the loop
    iter = 0
    while iter < maxiter
        iter += 1
        (predict_matrix, res_matrix) = (res_matrix, predict_matrix)
        (error, olderror) = (olderror, error)
        # transform vector into matrix
        fill!(res_matrix, y, ids.refs, times.refs)

        # principal components
        At_mul_B!(variance, res_matrix, res_matrix)
        F = eigfact!(Symmetric(variance), (length(times.pool) - rank + 1):length(times.pool))
        factors = F[:vectors]
        
        # predict matrix
        A_mul_Bt!(variance, factors, factors)
        A_mul_B!(predict_matrix, res_matrix, variance)

        # check convergence
        error = sqeuclidean(predict_matrix, res_matrix)
        if (error - olderror)/error < tol 
            converged[1] = true
            iterations[1] = iter
            break
        end
    end
    newfactors = Array(Float64, (length(times.pool), rank))
    for j in 1:rank
        newfactors[:, j] = factors[:, rank + 1 - j]
    end
    loadings = res_matrix * factors
    return PanelFactorResult(ids, times, loadings, newfactors, iterations, converged)
end


##############################################################################
##
## Estimate factor model by incremental backpropagation (Simon Funk Netflix Algorithm)
##
##############################################################################

function fit_backpropagation{Tids, Rids, Ttimes, Rtimes}(y::Vector{Float64}, id::PooledDataVector{Tids, Rids}, time::PooledDataVector{Ttimes, Rtimes}, rank::Integer, sqrtw::Vector{Float64} ; regularizer::Real = 0.0, learning_rate::Real = 1e-3, maxiter::Integer = 10000, tol::Real = 1e-9)
    
    # initialize
    idf = PooledFactor(id.refs, length(id.pool), rank)
    timef = PooledFactor(time.refs, length(time.pool), rank)
    iterations = fill(maxiter, rank)
    converged = fill(false, rank)
    for r in 1:rank
    	error = zero(Float64)
        olderror = zero(Float64)
        iter = 0
        while iter < maxiter
            iter += 1
            (error, olderror) = (olderror, error)
           	error = update!(idf, timef, y, sqrtw, regularizer, learning_rate / iter, r)
            # relative tolerance (absolute change would depend on learning_rate choice)
            if abs(error - olderror)/error < tol 
                iterations[r] = iter
                converged[r] = true
                break
            end
        end
        rescale!(idf, timef, r)
        subtract_factor!(y, sqrtw, idf, timef, r)
    end
    (loadings, factors) = rescale(idf.pool, timef.pool)

    PanelFactorResult(id, time, loadings, factors, iterations, converged)
end

