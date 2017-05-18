##############################################################################
##
## get weight
## 
##############################################################################

function get_weight(df::AbstractDataFrame, weight::Symbol) 
    out = df[weight]
    # there are no NA in it. DataVector to Vector
    out = convert(Vector{Float64}, out)
    map!(sqrt, out, out)
    return out
end
get_weight(df::AbstractDataFrame, ::Void) = Ones{Float64}(size(df, 1))



##############################################################################
##
## build model
##
##############################################################################

function reftype(sz) 
    sz <= typemax(UInt8)  ? UInt8 :
    sz <= typemax(UInt16) ? UInt16 :
    sz <= typemax(UInt32) ? UInt32 :
    UInt64
end
function ModelFrame2(trms::Terms, d::AbstractDataFrame, esample; contrasts::Dict = Dict())
    mf = ModelFrame(trms, d; contrasts = contrasts)
    mf.msng = esample
    return mf
end


#  remove observations with negative weights
function isnaorneg{T <: Real}(a::Vector{T}) 
    BitArray(a .> zero(eltype(a)))
end
function isnaorneg{T <: Real}(a::DataVector{T}) 
    out = .!(a.na)
    @inbounds @simd for i in 1:length(a)
        if out[i]
            out[i] = a[i] > zero(Float64)
        end
    end
    BitArray(out)
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


function _split(df::AbstractDataFrame, ss::Vector{Symbol})
    catvars, contvars = Symbol[], Symbol[]
    for s in ss
        isa(df[s], PooledDataVector) ? push!(catvars, s) : push!(contvars, s)
    end
    return catvars, contvars
end
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
        tss = sum(abs2, y)
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
        tss = sum(abs2, y)
    end
    return tss
end

