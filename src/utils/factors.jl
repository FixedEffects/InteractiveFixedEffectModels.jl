##############################################################################
##
## subtract_factor! and subtract_b!
##
##############################################################################

function subtract_factor!{R1, R2}(y::Vector{Float64}, sqrtw::AbstractVector{Float64}, idrefs::Vector{R1}, timerefs::Vector{R2}, idpool::Matrix{Float64}, timepool::Matrix{Float64})
    for r in 1:size(idpool, 2)
        subtract_factor!(y, sqrtw, idrefs, timerefs, slice(idpool, :, r), slice(timepool, :, r))
    end
end

function subtract_factor!{R1, R2}(y::Vector{Float64}, sqrtw::AbstractVector{Float64}, idrefs::Vector{R1}, timerefs::Vector{R2}, idpool::AbstractVector{Float64}, timepool::AbstractVector{Float64})
    @inbounds @simd for i in 1:length(y)
        y[i] -= sqrtw[i] * idpool[idrefs[i]] * timepool[timerefs[i]]
    end
end


##############################################################################
##
## compute sum of squared residuals
##
##############################################################################

function ssr{R1, R2}(y::Vector{Float64}, sqrtw::AbstractVector{Float64}, idrefs::Vector{R1}, timerefs::Vector{R2}, idpool::AbstractVector{Float64}, timepool::AbstractVector{Float64})
    out = zero(Float64)
    @inbounds @simd for i in 1:length(y)
        idi = idrefs[i]
        timei = timerefs[i]
        loading = idpool[idi]
        factor = timepool[timei]
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

function rescale!(idpool::AbstractVector{Float64}, timepool::AbstractVector{Float64})
    out = norm(timepool)
    @inbounds @simd for i in 1:length(idpool)
        idpool[i] *= out
    end
    invout = 1 / out
    @inbounds @simd for i in 1:length(timepool)
        timepool[i] *= invout
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

function DataFrame{R1, R2}(idrefs::Vector{R1}, timerefs::Vector{R2}, idpool::Matrix{Float64}, timepool::Matrix{Float64}, esample::BitVector)
    df = DataFrame()
    anyNA = all(esample)
    for r in 1:size(idpool, 2)
        # loadings
        df[convert(Symbol, "loadings$r")] = build_column(idrefs, idpool[:, r], esample)
        df[convert(Symbol, "factors$r")] = build_column(timerefs, timepool[:, r], esample)
    end
    return df
end


function build_column{R}(refs::Vector{R}, loadings::Vector{Float64}, esample::BitVector)
    T = eltype(refs)
    newrefs = fill(zero(T), length(esample))
    newrefs[esample] = refs
    return PooledDataArray(RefArray(newrefs), loadings)
end

