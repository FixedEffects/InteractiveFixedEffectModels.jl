##############################################################################
##
## Create light weight type
## 
##############################################################################

# http://stackoverflow.com/a/30968709/3662288
type Ones <: AbstractVector{Float64}
	length::Int
end
Base.size(O::Ones) = O.length
Base.getindex(O::Ones, I::Int...) = one(Float64)
Base.broadcast!(op::Function, X::Matrix{Float64}, Y::Matrix{Float64}, O::Ones) = nothing
Base.broadcast!(op::Function, X::Vector{Float64}, Y::Vector{Float64}, O::Ones) = nothing
Base.scale!(X::Vector{Float64}, O::Ones) = nothing


function get_weight(df::AbstractDataFrame, weight::Symbol)
	w = convert(Vector{Float64}, df[weight])
	sqrtw = sqrt(w)
end
get_weight(df::AbstractDataFrame, weight::Nothing) = Ones(size(df, 1))


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
## fill matrix into a vector and conversely
##
##############################################################################

function Base.fill!(M::Matrix{Float64}, x::Vector{Float64}, start::Integer)
	idx = start
	for i in 1:size(M, 1), j in 1:size(M, 2)
		idx += 1
		M[i, j] = x[idx]
	end
end

function Base.fill!(x::Vector{Float64}, M::Matrix{Float64}, start::Integer)
	idx = start
	for i in 1:size(M, 1), j in 1:size(M, 2)
		idx += 1
		x[idx] = M[i, j]
	end
end


function Base.fill!{Tid, Ttime}(ymatrix::Matrix{Float64}, yvector::Vector{Float64}, idsrefs::Vector{Tid}, timesrefs::Vector{Ttime})
	@inbounds @simd for i in 1:length(yvector)
		ymatrix[idsrefs[i], timesrefs[i]] = yvector[i]
	end
end

function Base.fill!{Tid, Ttime}(yvector::Vector{Float64}, ymatrix::Matrix{Float64},  idsrefs::Vector{Tid}, timesrefs::Vector{Ttime})
	@inbounds @simd for i in 1:length(yvector)
		yvector[i] = ymatrix[idsrefs[i], timesrefs[i]]
	end
end




##############################################################################
##
## build model
##
##############################################################################

function reftype(sz) 
	sz <= typemax(Uint8)  ? Uint8 :
	sz <= typemax(Uint16) ? Uint16 :
	sz <= typemax(Uint32) ? Uint32 :
	Uint64
end
function simpleModelFrame(df, t, esample)
	df1 = DataFrame(map(x -> df[x], t.eterms))
	names!(df1, convert(Vector{Symbol}, map(string, t.eterms)))
	mf = ModelFrame(df1, t, esample)
end


#  remove observations with negative weights
function isnaorneg{T <: Real}(a::Vector{T}) 
	bitpack(a .> zero(eltype(a)))
end
function isnaorneg{T <: Real}(a::DataVector{T}) 
	out = !a.na
	@inbounds @simd for i in 1:length(a)
		if out[i]
			out[i] = a[i] > zero(Float64)
		end
	end
	bitpack(out)
end


# Directly from DataFrames.jl
function remove_response(t::Terms)
    # shallow copy original terms
    t = Terms(t.terms, t.eterms, t.factors, t.order, t.response, t.intercept)
    if t.response
        t.order = t.order[2:end]
        t.eterms = t.eterms[2:end]
        t.factors = t.factors[2:end, 2:end]
        t.response = false
    end
    return t
end


# used when removing certain rows in a dataset
# NA always removed
function dropUnusedLevels!(f::PooledDataVector)
	uu = unique(f.refs)
	length(uu) == length(f.pool) && return f
	sort!(uu)
	T = reftype(length(uu))
	dict = Dict(uu, 1:convert(T, length(uu)))
	@inbounds @simd for i in 1:length(f.refs)
		 f.refs[i] = dict[f.refs[i]]
	end
	f.pool = f.pool[uu]
	f
end

dropUnusedLevels!(f::DataVector) = f



function build_column(id::PooledDataVector, loadings::Matrix{Float64}, r::Int, esample::BitVector)
	T = eltype(id.refs)
	refs = fill(zero(T), length(esample))
	refs[esample] = id.refs
	return PooledDataArray(RefArray(refs), loadings[:, r])
end


##############################################################################
##
## sum of squares
##
##############################################################################



function compute_ss(residuals::Vector{Float64}, y::Vector{Float64}, hasintercept::Bool, sqrtw::Ones)
	ess = abs2(norm(residuals))
	if hasintercept
		tss = zero(Float64)
		m = mean(y)::Float64
		@inbounds @simd  for i in 1:length(y)
			tss += abs2((y[i] - m))
		end
	else
		tss = abs2(norm(y))
	end
	return (ess, tss)
end
function compute_ss(residuals::Vector{Float64}, y::Vector{Float64}, hasintercept::Bool, sqrtw::Vector{Float64})
	ess = abs2(norm(residuals))
	if hasintercept
		m = (mean(y) / sum(sqrtw) * length(residuals))::Float64
		tss = zero(Float64)
		@inbounds @simd  for i in 1:length(y)
		 tss += abs2(y[i] - sqrtw[i] * m)
		end
	else
		tss = abs2(norm(y))
	end
	return (ess, tss)
end


