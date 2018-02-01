##############################################################################
##
## get weight
## 
##############################################################################

function get_weights(df::AbstractDataFrame, weights::Symbol) 
    out = df[weights]
    # there are no missing in it
    out = convert(Vector{Float64}, out)
    map!(sqrt, out, out)
    return out
end
get_weights(df::AbstractDataFrame, ::Void) = Ones{Float64}(size(df, 1))


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
function isnaorneg(a::Vector{T}) where {T}
    out = BitArray(length(a))
    @simd for i in 1:length(a)
        @inbounds out[i] = !ismissing(a[i]) && (a[i] > zero(T))
    end
    return out
end


function _split(df::AbstractDataFrame, ss::Vector{Symbol})
    catvars, contvars = Symbol[], Symbol[]
    for s in ss
        isa(df[s], CategoricalVector) ? push!(catvars, s) : push!(contvars, s)
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

