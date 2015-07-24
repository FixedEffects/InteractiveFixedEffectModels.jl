
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
## read formula
##
##############################################################################


# decompose formula into normal + iv vs absorbpart
function decompose_absorb!(rf::Formula)
	has_absorb = false
	absorb_formula = nothing
	if typeof(rf.rhs) == Expr && rf.rhs.args[1] == :(|>)
		has_absorb = true
		absorb_formula = Formula(nothing, rf.rhs.args[3])
		rf.rhs = rf.rhs.args[2]
	end
	return(rf, has_absorb, absorb_formula)
end



##############################################################################
##
## build model
##
##############################################################################


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


function allvars(ex::Expr)
    if ex.head != :call error("Non-call expression encountered") end
    [[allvars(a) for a in ex.args[2:end]]...]
end
allvars(f::Formula) = unique(vcat(allvars(f.rhs), allvars(f.lhs)))
allvars(sym::Symbol) = [sym]
allvars(v::Any) = Array(Symbol, 0)


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


