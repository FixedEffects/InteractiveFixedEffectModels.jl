# type used internally to store idrefs, timerefs, factors, loadings
type PooledFactor{R}
    refs::Vector{R}
    pool::Matrix{Float64}
    old1pool::Matrix{Float64}
    old2pool::Matrix{Float64}
    x::Vector{Float64}
    x_ls::Vector{Float64}
    gr::Vector{Float64}
    gr_ls::Vector{Float64}
end

function PooledFactor{R}(refs::Vector{R}, l::Integer, rank::Integer)
    ans = fill(zero(Float64), l)
    PooledFactor(refs, fill(0.1, l, rank), fill(0.1, l, rank), fill(0.1, l, rank), ans, deepcopy(ans), deepcopy(ans), deepcopy(ans))
end

##############################################################################
##
## subtract_factor! and subtract_b!
##
##############################################################################

function subtract_b!(y::Vector{Float64}, b::Vector{Float64}, X::Matrix{Float64})
    BLAS.gemm!('N', 'N', -1.0, X, b, 1.0, y)
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
   return out
end

##############################################################################
##
## rescale! a factor model
##
##############################################################################
# reverse columns in a matrix
function reverse{R}(m::Matrix{R})
    out = similar(m)
    for j in 1:size(m, 2)
        invj = size(m, 2) + 1 - j 
        @inbounds @simd for i in 1:size(m, 1)
            out[i, j] = m[i, invj]
        end
    end
    return out
end

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


##############################################################################
##
## DataFrame from factors loadings
##
##############################################################################

function getfactors{Rid, Rtime}(y::Vector{Float64},
                                id::PooledFactor{Rid},
                                time::PooledFactor{Rtime},
                                sqrtw::AbstractVector{Float64})

    # partial out Y and X with respect to i.id x factors and i.time x loadings
    newfes = FixedEffect[]
    ans = Array(Float64, length(y))
    for j in 1:size(id.pool, 2)
        for i in 1:length(y)
            ans[i] = time.pool[time.refs[i], j]
        end
        currentid = FixedEffect(id.refs, size(id.pool, 1), sqrtw, ans[:], :id, :time, :(idxtime))
        push!(newfes, currentid)
        for i in 1:length(y)
            ans[i] = id.pool[id.refs[i], j]
        end
        currenttime = FixedEffect(time.refs, size(time.pool, 1), sqrtw, ans[:], :time, :id, :(timexid))
        push!(newfes, currenttime)
    end
    # obtain the residuals and cross 
    return newfes
end