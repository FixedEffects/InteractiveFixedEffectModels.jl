
##############################################################################
##
## update! by gauss-seidel method
##
##############################################################################

function update!{R1, R2}(id::PooledFactor{R1}, time::PooledFactor{R2}, y::Vector{Float64}, sqrtw::AbstractVector{Float64}, r::Integer)
	changeid = update_half!(id, time, y, sqrtw, r)
	changetime = update_half!(time, id, y, sqrtw, r)
	return max(changeid,changetime)
end

function update_half!{R1, R2}(p1::PooledFactor{R1}, p2::PooledFactor{R2}, y::Vector{Float64}, sqrtw::AbstractVector{Float64}, r::Integer)
	fill!(p1.storage1, zero(Float64))
	fill!(p1.storage2, zero(Float64))
	 @inbounds @simd for i in 1:length(p1.refs)
		 pi = p1.refs[i]
		 yi = y[i]
		 xi = sqrtw[i] * p2.pool[p2.refs[i], r] 
		 p1.storage1[pi] += xi * yi
		 p1.storage2[pi] += abs2(xi)
	end
	change = zero(Float64)
	 @inbounds @simd for i in 1:size(p1.pool, 1)
		if p1.storage2[i] > zero(Float64)
			result = p1.storage1[i] / p1.storage2[i]
			current = abs(p1.pool[i, r] - result)
			if current > change
				change = current
			end
			p1.pool[i, r] = result
		end
	end
	return change
end


##############################################################################
##
## update! by backpropagation
##
##############################################################################


function update!{R1, R2}(id::PooledFactor{R1}, time::PooledFactor{R2}, y::Vector{Float64}, sqrtw::AbstractVector{Float64}, regularizer::Real, learning_rate::Real, r::Integer)
	out = zero(Float64)
     @inbounds @simd for i in 1:length(y)
        idi = id.refs[i]
        timei = time.refs[i]
        loading = id.pool[idi, r]
        factor = time.pool[timei, r]
        sqrtwi = sqrtw[i]
        error = y[i] - sqrtwi * loading * factor 
        out += abs2(error)
        id.pool[idi, r] += learning_rate * 2.0 * (error * sqrtwi * factor - regularizer * loading)
        time.pool[timei, r] += learning_rate * 2.0 * (error * sqrtwi * loading - regularizer * factor)
    end
    return out
end



##############################################################################
##
## rescale! a factor model
##
##############################################################################

function rescale!{R1, R2}(id::PooledFactor{R1}, time::PooledFactor{R2}, r::Integer)
	out = zero(Float64)
	 @inbounds @simd for i in 1:size(time.pool, 1)
		out += abs2(time.pool[i, r])
	end
	out = sqrt(out)
	 @inbounds @simd for i in 1:size(id.pool, 1)
		id.pool[i, r] *= out
	end
	invout = 1 / out
	 @inbounds @simd for i in 1:size(time.pool, 1)
		time.pool[i, r] *= invout
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
## subtract_factor! and subtract_b!
##
##############################################################################

function subtract_b!(res::Vector{Float64}, y::Vector{Float64}, b::Vector{Float64}, X::Matrix{Float64})
	A_mul_B!(res, X, -b)
	 @inbounds @simd for i in 1:length(res)
		res[i] += y[i] 
	end
end

#To do : define with abstract Vector
function subtract_factor!{R1, R2}(y::Vector{Float64}, sqrtw::AbstractVector{Float64}, idrefs::Vector{R1}, loadings::Matrix{Float64}, timerefs::Vector{R2}, factors::Matrix{Float64}, r::Integer)
	 @inbounds @simd for i in 1:length(y)
		y[i] -= sqrtw[i] * loadings[idrefs[i], r] * factors[timerefs[i], r]
	end
end

function subtract_factor!{R1, R2}(y::Vector{Float64}, sqrtw::AbstractVector{Float64},  id::PooledFactor{R1}, time::PooledFactor{R2}, r::Integer)
	subtract_factor!(y, sqrtw, id.refs, id.pool, time.refs, time.pool, r)
end

function subtract_factor!{R1, R2}(res::Vector{Float64}, y::Vector{Float64}, sqrtw::AbstractVector{Float64},  p1::PooledFactor{R1}, p2::PooledFactor{R2})
	 @inbounds @simd for i in 1:length(y)
		res[i] = y[i] - sqrtw[i] * p1.pool[p1.refs[i], 1] * p2.pool[p2.refs[i], 1]
	end
	for r in 2:size(p1.pool, 2)
		subtract_factor!(res, sqrtw, p1, p2, r)
	end
end






