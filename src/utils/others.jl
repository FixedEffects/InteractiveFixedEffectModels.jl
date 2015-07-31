
##############################################################################
##
## Reverse matrix (eifact! gives smaller eigenvalues first)
## 
##############################################################################

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


##############################################################################
##
## Chebyshev
##
##############################################################################

function chebyshev{T, N}(x::Array{T, N})
    distance = zero(Float64)
    @inbounds @simd for i in eachindex(x)
        current = abs(x[i])
        if current > distance
            distance = current
        end
    end
    return distance
end

function chebyshev{T, N}(x::Array{T, N}, y::Array{T, N})
    distance = zero(Float64)
    @inbounds @simd for i in eachindex(x)
        current = abs(x[i] - y[i])
        if current > distance
            distance = current
        end
    end
    return distance
end


function sqeuclidean{T, N}(x::Array{T, N}, y::Array{T, N})
    distance = zero(Float64)
    @inbounds @simd for i in eachindex(x)
        distance += abs2(x[i] - y[i])
    end
    return sqrt(distance)
end



##############################################################################
##
## some methods to copy! bjects
##
##############################################################################

function Base.copy!(M::Matrix{Float64}, x::Vector{Float64}, start::Integer)
    idx = start
    for i in 1:size(M, 1), j in 1:size(M, 2)
        idx += 1
        M[i, j] = x[idx]
    end
end

function Base.copy!(x::Vector{Float64}, M::Matrix{Float64}, start::Integer)
    idx = start
    for i in 1:size(M, 1), j in 1:size(M, 2)
        idx += 1
        x[idx] = M[i, j]
    end
end


function Base.copy!{Tid, Ttime}(ymatrix::Matrix{Float64}, yvector::Vector{Float64}, idsrefs::Vector{Tid}, timesrefs::Vector{Ttime})
    @inbounds @simd for i in 1:length(yvector)
        ymatrix[idsrefs[i], timesrefs[i]] = yvector[i]
    end
end

function Base.copy!{Tid, Ttime}(yvector::Vector{Float64}, ymatrix::Matrix{Float64},  idsrefs::Vector{Tid}, timesrefs::Vector{Ttime})
    @inbounds @simd for i in 1:length(yvector)
        yvector[i] = ymatrix[idsrefs[i], timesrefs[i]]
    end
end

function Base.copy!(pool::Matrix{Float64}, store::Vector{Float64}, r::Int)
    @inbounds @simd for i in 1:size(pool, 1)
        pool[i, r] = store[i]
    end
end

function Base.copy!(store::Vector{Float64}, pool::Matrix{Float64}, r::Int)
    @inbounds @simd for i in 1:size(pool, 1)
        store[i] = pool[i, r]
    end
end

function Base.copy!(pool::Matrix{Float64}, store::Matrix{Float64}, r::Int)
    @inbounds @simd for i in 1:size(pool, 1)
        pool[i, r] = store[i, r]
    end
end

function Base.copy!(store::Matrix{Float64}, pool::Matrix{Float64}, r::Int)
    @inbounds @simd for i in 1:size(pool, 1)
        store[i, r] = pool[i, r]
    end
end




