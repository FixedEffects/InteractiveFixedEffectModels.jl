

##############################################################################
##
## compute sum of squared residuals
##
##############################################################################

function ssr{R1, R2}(id::PooledFactor{R1}, time::PooledFactor{R2}, y::Vector{Float64}, sqrtw::AbstractVector{Float64}, r::Integer, lambda::Real = 0.0)
    out = zero(Float64)
    @inbounds @simd for i in 1:length(y)
        idi = id.refs[i]
        timei = time.refs[i]
        loading = id.pool[idi, r]
        factor = time.pool[timei, r]
        sqrtwi = sqrtw[i]
        out += abs2(y[i] - sqrtwi * loading * factor)
    end 
    if lambda > 0.0
        @inbounds @simd for i in 1:size(id.pool, 1)
            out += lambda * abs2(id.pool[i, r])
        end
        @inbounds @simd for i in 1:size(time.pool, 1)
            out += lambda * abs2(time.pool[i, r])
        end
    end
    return out
end



function ssr{R1, R2}(id::PooledFactor{R1}, time::PooledFactor{R2}, b::Vector{Float64}, Xt::Matrix{Float64}, y::Vector{Float64}, sqrtw::AbstractVector{Float64}, lambda::Real = 0.0)    
    out = zero(Float64)
    n_regressors = length(b)
    rank = size(id.pool, 2)
    @inbounds @simd for i in 1:length(y)
        prediction = zero(Float64)
        idi = id.refs[i]
        timei = time.refs[i]
        sqrtwi = sqrtw[i]
        for k in 1:n_regressors
            prediction += b[k] * Xt[k, i]
        end
        for r in 1:rank
          prediction += sqrtwi * id.pool[idi, r] * time.pool[timei, r]
        end
        error = y[i] - prediction
        out += abs2(error)
    end
    if lambda > 0.0
        for r in 1:rank
            @inbounds @simd for i in 1:size(id.pool, 1)
                out += lambda * abs2(id.pool[i, r])
            end
            @inbounds @simd for i in 1:size(time.pool, 1)
                out += lambda * abs2(time.pool[i, r])
            end
        end
    end
    return out
end



##############################################################################
##
## subtract_factor! and subtract_b!
##
##############################################################################

function subtract_b!(res::Vector{Float64}, y::Vector{Float64}, b::Vector{Float64}, X::Matrix{Float64})
    A_mul_B!(res, X, -b)
     @inbounds @simd for i in 1:length(res)
        res[i] += y[i] 
    end
end


function subtract_factor!{R1, R2}(y::Vector{Float64}, sqrtw::AbstractVector{Float64}, id::PooledFactor{R1}, time::PooledFactor{R2})
    for r in 1:size(id.pool, 2)
        subtract_factor!(y, sqrtw, id, time, r)
    end
end

function subtract_factor!{R1, R2}(y::Vector{Float64}, sqrtw::AbstractVector{Float64}, id::PooledFactor{R1}, time::PooledFactor{R2}, r::Integer)
     @inbounds @simd for i in 1:length(y)
        y[i] -= sqrtw[i] * id.pool[id.refs[i], r] * time.pool[time.refs[i], r]
    end
end



##############################################################################
##
## rescale! a factor model
##
##############################################################################


function rescale!{R1, R2}(id::PooledFactor{R1}, time::PooledFactor{R2}, r::Integer)
	out = zero(Float64)
	 @inbounds @simd for i in 1:size(time.pool, 1)
		out += abs2(time.pool[i, r])
	end
	out = sqrt(out)
	 @inbounds @simd for i in 1:size(id.pool, 1)
		id.pool[i, r] *= out
	end
	invout = 1 / out
	 @inbounds @simd for i in 1:size(time.pool, 1)
		time.pool[i, r] *= invout
	end
end

# normalize factors and loadings so that F'F = Id, Lambda'Lambda diagonal
function rescale!(scaledloadings::Matrix{Float64}, scaledfactors::Matrix{Float64}, loadings::Matrix{Float64}, factors::Matrix{Float64})
    U = eigfact!(Symmetric(At_mul_B(factors, factors)))
    sqrtDx = diagm(sqrt(abs(U[:values])))
    A_mul_B!(scaledloadings,  loadings,  U[:vectors] * sqrtDx)
    V = eigfact!(At_mul_B(scaledloadings, scaledloadings))
    A_mul_B!(scaledloadings, loadings, reverse(U[:vectors] * sqrtDx * V[:vectors]))
    A_mul_B!(scaledfactors, factors, reverse(U[:vectors] * (sqrtDx \ V[:vectors])))
    return scaledloadings, scaledfactors
end

function rescale(loadings::Matrix{Float64}, factors::Matrix{Float64})
	scaledloadings = similar(loadings)
	scaledfactors = similar(factors)
	rescale!(scaledloadings, scaledfactors, loadings, factors)
	return scaledloadings, scaledfactors
end


##############################################################################
##
## Create dataframe from pooledfactors
##
##############################################################################


function DataFrame(id::PooledFactor, time::PooledFactor, esample::BitVector)
    df = DataFrame()
    anyNA = all(esample)
    for r in 1:size(id.pool, 2)
        # loadings
        df[convert(Symbol, "loadings$r")] = build_column(id.refs, id.pool, r, esample)
        df[convert(Symbol, "factors$r")] = build_column(time.refs, time.pool, r, esample)
    end
    return df
end


function build_column{R}(refs::Vector{R}, loadings::Matrix{Float64}, r::Int, esample::BitVector)
    T = eltype(refs)
    newrefs = fill(zero(T), length(esample))
    newrefs[esample] = refs
    return PooledDataArray(RefArray(newrefs), loadings[:, r])
end

