
##############################################################################
##
## update! by alternative regressions
##
##############################################################################

function update_ar!{R1, R2}(id::PooledFactor{R1}, time::PooledFactor{R2}, y::Vector{Float64}, sqrtw::AbstractVector{Float64}, r::Integer)
	update_ar_half!(id, time, y, sqrtw, r)
	update_ar_half!(time, id, y, sqrtw, r)
end

function update_ar_half!{R1, R2}(p1::PooledFactor{R1}, p2::PooledFactor{R2}, y::Vector{Float64}, sqrtw::AbstractVector{Float64}, r::Integer)
	fill!(p1.storage1, zero(Float64))
	fill!(p1.storage2, zero(Float64))
	 @inbounds @simd for i in 1:length(p1.refs)
		 pi = p1.refs[i]
		 yi = y[i]
		 xi = sqrtw[i] * p2.pool[p2.refs[i], r] 
		 p1.storage1[pi] += xi * yi
		 p1.storage2[pi] += abs2(xi)
	end
	 @inbounds @simd for i in 1:size(p1.pool, 1)
		if p1.storage2[i] > zero(Float64)
			p1.pool[i, r] = p1.storage1[i] / p1.storage2[i]
		end
	end
end


##############################################################################
##
## update! by gradient method
##
##############################################################################

function update_gd!{R1, R2}(id::PooledFactor{R1}, time::PooledFactor{R2}, y::Vector{Float64}, sqrtw::AbstractVector{Float64}, r::Integer, learning_rate::Float64, lambda::Float64)
    update_gd_half!(id, time, y, sqrtw, r, learning_rate, lambda)
    update_gd_half!(time, id, y, sqrtw, r, learning_rate, lambda)
end

function update_gd_half!{R1, R2}(p1::PooledFactor{R1}, p2::PooledFactor{R2}, y::Vector{Float64}, sqrtw::AbstractVector{Float64}, r::Integer, learning_rate::Float64, lambda::Float64)
    fill!(p1.storage1, zero(Float64))
    fill!(p1.storage2, zero(Float64))
    @inbounds @simd for i in 1:length(y)
        idi = p1.refs[i]
        timei = p2.refs[i]
        loading = p1.pool[idi, r]
        factor = p2.pool[timei, r]
        sqrtwi = sqrtw[i]
        error = y[i] - sqrtwi * loading * factor 
        p1.storage1[idi] += 2.0 * (error * sqrtwi * factor)
        p1.storage2[idi] += abs2(factor)
    end

    @inbounds @simd for i in 1:size(p1.pool, 1)
        p1.pool[i, r] -= 2.0 * lambda * p1.pool[i, r]
        if p1.storage2[i] > zero(Float64)
            # gradient term is learning rate * p1.storage2
            # newton term (diagonal only) is p1.storage2[i].
            # momentum term. mu = 0 actually makes things faster for factor model
            p1.pool[i, r] += learning_rate * p1.storage1[i] / p1.storage2[i]^0.5 + 0.00 * (p1.old1pool[i, r] - p1.old2pool[i, r])
        end
    end
end


##############################################################################
##
## update! by stochastic gradient
##
##############################################################################

function update_sgd!{R1, R2}(id::PooledFactor{R1}, time::PooledFactor{R2}, y::Vector{Float64}, sqrtw::AbstractVector{Float64}, r::Int, learning_rate::Real, lambda::Real, range::UnitRange{Int})
     @inbounds for j in 1:length(y)
        i = rand(range)
        idi = id.refs[i]
        timei = time.refs[i]
        loading = id.pool[idi, r]
        factor = time.pool[timei, r]
        sqrtwi = sqrtw[i]
        error = y[i] - sqrtwi * loading * factor 
        id.pool[idi, r] += learning_rate * 2.0 * (error * sqrtwi * factor - lambda * loading)
        time.pool[timei, r] += learning_rate * 2.0 * (error * sqrtwi * loading - lambda * factor)
    end   
end