function tss(y::AbstractVector, hasintercept::Bool, weights::AbstractWeights)
    if hasintercept
        m = mean(y, weights)
        sum(@inbounds (y[i] - m)^2 * weights[i] for i in eachindex(y))
    else
        sum(@inbounds y[i]^2 * weights[i] for i in eachindex(y))
    end
end