##############################################################################
##
## light weight type
## 
##############################################################################

# http://stackoverflow.com/a/30968709/3662288
type Ones <: AbstractVector{Float64}
    length::Int
end
Base.size(O::Ones) = O.length
Base.getindex(::Ones, ::Int...) = one(Float64)
Base.broadcast!(::Function, ::Matrix{Float64}, ::Matrix{Float64}, ::Ones) = nothing
Base.broadcast!(::Function, ::Vector{Float64}, ::Vector{Float64}, ::Ones) = nothing
Base.scale!(::Vector{Float64}, ::Ones) = nothing


function get_weight(df::AbstractDataFrame, weight::Symbol)
    w = convert(Vector{Float64}, df[weight])
    sqrtw = sqrt(w)
end
get_weight(df::AbstractDataFrame, ::Nothing) = Ones(size(df, 1))

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
    dict = Dict{eltype(uu), T}(zip(uu, collect(1:convert(T, length(uu)))))
    @inbounds @simd for i in 1:length(f.refs)
         f.refs[i] = dict[f.refs[i]]
    end
    f.pool = f.pool[uu]
    f
end

dropUnusedLevels!(f::DataVector) = f

##############################################################################
##
## sum of squares
##
##############################################################################

function compute_tss(y::Vector{Float64}, hasintercept::Bool, ::Ones)
    if hasintercept
        tss = zero(Float64)
        m = mean(y)::Float64
        @inbounds @simd  for i in 1:length(y)
            tss += abs2((y[i] - m))
        end
    else
        tss = sumabs2(y)
    end
    return tss
end

function compute_tss(y::Vector{Float64}, hasintercept::Bool, sqrtw::Vector{Float64})
    if hasintercept
        m = (mean(y) / sum(sqrtw) * length(y))::Float64
        tss = zero(Float64)
        @inbounds @simd  for i in 1:length(y)
         tss += abs2(y[i] - sqrtw[i] * m)
        end
    else
        tss = sumabs2(y)
    end
    return tss
end

