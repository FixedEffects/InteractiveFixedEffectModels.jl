##############################################################################
##
## Factor Models
##
##############################################################################
function fit(m::PanelFactorModel, variable::Symbol, df::AbstractDataFrame; subset::Union(AbstractVector{Bool}, Nothing) = nothing, weight::Union(Symbol, Nothing) = nothing, method::Symbol = :gradient_descent, lambda::Real = 0.0)

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
    ids = subdf[m.id]
    times = subdf[m.time]

    # initialize
    iterations = Int[]
    iteration_converged = Bool[]
    x_converged = Bool[]
    f_converged = Bool[]
    gr_converged = Bool[]

    loadings =  Array(Float64, length(ids.pool))
    factors =  Array(Float64, length(times.pool))
    loadingsmatrix = Array(Float64, (length(ids.pool), m.rank))
    factorsmatrix = Array(Float64, (length(times.pool), m.rank))

    # x is a vector of length N + T, whigh holds loading_n for n < N and factor_{n-N} for n > N
    # this dataformat is required by optimize in Optim.jl
    l = length(ids.pool)
    x0 = fill(0.1, length(ids.pool) + length(times.pool))

    for r in 1:m.rank
        d = DifferentiableFunction(x -> sum_of_squares(x, sqrtw, y, times.refs, ids.refs, l, lambda),
                                    (x, storage) -> sum_of_squares_gradient!(x, storage, sqrtw, y, times.refs, ids.refs, l, lambda),
                                    (x, storage) -> sum_of_squares_and_gradient!(x, storage, sqrtw, y, times.refs, ids.refs, l, lambda)
                                    )
        # optimize
        result = optimize(d, x0, method = method)  

        # write results
        loadings[:] = result.minimum[1:l]
        factors[:] = result.minimum[(l+1):end]
        push!(iterations, result.iterations)
        push!(iteration_converged, result.iteration_converged)
        push!(x_converged, result.x_converged)
        push!(f_converged, result.f_converged)
        push!(gr_converged, result.gr_converged)
        
        # residualize
        residualize!(y, factors, loadings, ids.refs, times.refs)
       
        # normalize so that F'F/T = Id
        factorscale = norm(factors, 2) / sqrt(length(times.pool))
        scale!(factors, 1 / factorscale)
        scale!(loadings, factorscale)

        factorsmatrix[:, r] = factors
        loadingsmatrix[:, r] = loadings
    end

    PanelFactorResult(ids, times, loadingsmatrix, factorsmatrix, iterations, iteration_converged, x_converged, f_converged, gr_converged)
end

# function to minimize
function sum_of_squares{Ttime, Tid}(x::Vector{Float64}, sqrtw::Vector{Float64}, y::Vector{Float64}, timesrefs::Vector{Ttime}, idsrefs::Vector{Tid}, l::Integer, lambda::Real)
    out = zero(Float64)
    @inbounds @simd for i in 1:length(y)
        id = idsrefs[i]
        time = timesrefs[i] + l
        loading = x[id]
        factor = x[time]
        sqrtwi = sqrtw[i]
        error = sqrtwi * (y[i] - loading * factor)
        out += abs2(error)
    end
    # penalty term
    @inbounds @simd for i in 1:length(x)
        out += lambda * abs2(x[i])
    end
    return out
end

# gradient
function sum_of_squares_gradient!{Ttime, Tid}(x::Vector{Float64}, storage::Vector{Float64}, sqrtw::Vector{Float64}, y::Vector{Float64}, timesrefs::Vector{Ttime}, idsrefs::Vector{Tid}, l::Integer, lambda::Real)
    fill!(storage, zero(Float64))
    @inbounds @simd for i in 1:length(y)
        id = idsrefs[i]
        time = timesrefs[i] + l
        loading = x[id]
        factor = x[time]
        sqrtwi = sqrtw[i]
        error = sqrtwi * (y[i] - loading * factor)
        storage[id] -= 2.0 * error * sqrtwi * factor 
        storage[time] -= 2.0 * error * sqrtwi * loading
    end
    # penalty term
    @inbounds @simd for i in 1:length(x)
        storage[i] += 2.0 * lambda * x[i]
    end
    return storage
end

# function + gradient in the same loop
function sum_of_squares_and_gradient!{Ttime, Tid}(x::Vector{Float64}, storage::Vector{Float64}, sqrtw::Vector{Float64}, y::Vector{Float64}, timesrefs::Vector{Ttime}, idsrefs::Vector{Tid}, l::Integer, lambda::Real)
    fill!(storage, zero(Float64))
    out = zero(Float64)
    @inbounds @simd for i in 1:length(y)
        id = idsrefs[i]
        time = timesrefs[i]+l
        loading = x[id]
        factor = x[time]
        sqrtwi = sqrtw[i]
        error =  sqrtwi * (y[i] - loading * factor)
        out += abs2(error)
        storage[id] -= 2.0 * error * sqrtwi * factor 
        storage[time] -= 2.0 * error * sqrtwi * loading 
    end
    # penalty term
    @inbounds @simd for i in 1:length(x)
        out += lambda * abs2(x[i])
        storage[i] += 2.0 * lambda * x[i]
    end
    return out
end



function residualize!{Ttime, Tid}(y::Vector{Float64}, factors::Vector{Float64}, loadings::Vector{Float64}, idsrefs::Vector{Tid}, timesrefs::Vector{Ttime})
    @inbounds @simd for i in 1:length(y)
        y[i] -= factors[timesrefs[i]] * loadings[idsrefs[i]]
    end
end









