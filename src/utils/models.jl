##############################################################################
##
## get weight
## 
##############################################################################

function get_weights(df::AbstractDataFrame, esample::AbstractVector, weights::Symbol) 
    # there are no NA in it. DataVector to Vector
    out = convert(Vector{Float64}, df[esample, weights])
    map!(sqrt, out, out)
    return out
end
get_weights(df::AbstractDataFrame, esample::AbstractVector, ::Nothing) = Ones{Float64}(sum(esample))

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




#  remove observations with negative weights
isnaorneg(a::AbstractVector) = BitArray(!ismissing(x) & (x > 0) for x in a)


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

