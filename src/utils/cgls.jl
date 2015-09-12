# This solves Ax = b
# r should equal b - Ax0 where x0 is an initial guess for x. It is NOT modified in place
# x, s, p, q are used for storage. s, p should have dimension size(A, 2). q should have simension size(A, 1). 

# TODO. Follow LMQR for (i) better stopping rule (ii) better projection on zero in case x non identified
function cgls!(x::Union(AbstractVector{Float64}, Nothing), 
               r::AbstractVector{Float64}, A::AbstractMatrix{Float64}, 
               normalization::AbstractVector{Float64}, s::AbstractVector{Float64}, 
               z::AbstractVector{Float64}, p::AbstractVector{Float64}, 
               q::AbstractVector{Float64}, ptmp; 
               tol::Real=1e-5, maxiter::Int=10)

    # Initialization.
    converged = false
    iterations = maxiter 

    Ac_mul_B!(s, A, r)
    sumabs2!(normalization, A)
    broadcast!(/, z, s, normalization)
    copy!(p, z)
    normS0 = dot(s, z)
    normSold = normS0  

    iter = 0
    while iter < maxiter
        iter += 1
        A_mul_B!(q, A, p) 
        Ac_mul_B!(ptmp, A, q)
        α = normSold / dot(ptmp, p)
        # x = x + αp
        x == nothing || axpy!(α, p, x) 
        axpy!(-α, ptmp, s)
        broadcast!(/, z, s, normalization)
        normS = dot(s, z)
        if α * maxabs(q) <= tol 
            iterations = iter
            converged = true
            break
        end
        β = normS / normSold
        # p = s + β p
        scale!(p, β)
        axpy!(1.0, z, p) 
        normSold = normS
    end
    return iterations, converged
end


function dogleg!(x, fg, fcur, f!, g!; tol =1e-8, maxiter=1000, lambda=0.0)
    Δ = 1.0
    g = similar(x)
    δgn = similar(x) # gauss newton step
    δsd = similar(x) # steepest descent
    δdiff = similar(x) # δgn - δsd
    δx = similar(x)
    ftrial = similar(fcur)
    ftmp = similar(fcur)
    q = similar(fcur)

    # for cgls
    s = similar(x)
    p = similar(x)
    q = similar(fcur)
    z = similar(s)
    ptmp = similar(x)
    ptmp2 = similar(x)
    normalization = similar(x)

    f!(x, fcur)
    g!(x, fg)
    d = similar(x)
    sumabs2!(d, fg)
    scale!(fcur, -1.0)
    residual = sumabs2(fcur)

    iter = 0
    while iter < maxiter 
        iter += 1
        g!(x, fg)
        Ac_mul_B!(g, fg, fcur)
        broadcast!(/, g, g, d)
        if maxabs(g) <= tol
            return iter, true
        end
        A_mul_B!(ftmp, fg, g)
        scale!(δsd, g, wsumabs2(d, g)/sumabs2(ftmp))
        gncomputed = false
        ρ = -1.0
        while ρ < 0
            # compute δx
            if wsumabs2(d, δsd) >= Δ^2
                # Cauchy point is out of the region
                # take largest Cauchy step withub the trust region boundary
                scale!(δx, δsd, Δ/sqrt(wsumabs2(d, δsd)))
            else
                if (!gncomputed)
                    fill!(δgn, zero(Float64))
                    cgls!(δgn, fcur, fg, normalization, s, z, p, q, ptmp)
                    # δdiff = δgn - δsd
                    copy!(δdiff, δgn)
                    axpy!(-1.0, δsd,  δdiff)
                    gncomputed = true
                end
                if wsumabs2(d, δgn) <= Δ^2
                    # Gauss-Newton step is within the region
                    copy!(δx, δgn)
                else
                    # Gauss-Newton step is outside the region
                    # intersection trust region and line Cauchy point and the Gauss-Newton step
                    b = 2 * wdot(d, δsd, δdiff)
                    a = wsumabs2(d, δdiff)
                    c = wsumabs2(d, δsd)
                    tau = (-b+sqrt(b^2-4*a*(c-Δ^2)))/2/a 
                    copy!(δx, δsd)
                    axpy!(tau, δdiff, δx)
                end
            end

            # update x
            if norm(δx) <= tol * norm(x)
                return iter, true
            else           
                axpy!(1.0, δx, x)
                f!(x, ftrial)
                out = predicted_value!(ftmp, fcur, fg, δx)
                trial_residual = sumabs2(ftrial)
                ρ = (trial_residual - residual)/(out-residual)
                if ρ > 0
                    # Successful iteration
                    copy!(fcur, ftrial)
                    scale!(fcur, -1.0)
                    g!(x, fg)
                    sumabs2!(d, fg)
                    residual = trial_residual
                else
                    axpy!(-1.0, δx, x)
                end
                if ρ < 0.25
                   Δ = Δ / 2
                elseif ρ > 0.75
                   Δ = 2 * Δ
               end
                if Δ <= tol * norm(x)
                    return iter, true
                end
            end
        end
    end
    return maxiter, false
end


function predicted_value!(ftmp, fcur, fg, δx)
    A_mul_B!(ftmp, fg, δx)
    # Ratio of actual to predicted reduction
    out = zero(Float64)
    @inbounds @simd for i in 1:length(fcur)
        out += abs2(-fcur[i] + ftmp[i])
    end
    return out
end

function wdot(d, x, y)
    out = zero(Float64)
    @inbounds @simd for i in 1:length(x)
        out += d[i] * x[i] * y[i]
    end
    return out
end

function wsumabs2(d, x)
    out = zero(Float64)
    @inbounds @simd for i in 1:length(x)
        out += d[i] * x[i]^2
    end
    return out
end

