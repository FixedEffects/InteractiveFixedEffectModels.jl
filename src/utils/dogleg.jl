##############################################################################
##
## dogleg method
## Is Levenberg-Marquardt the Most Efficient Optimization Algorithm for Implementing Bundle Adjustment? ## Manolis I.A. Lourakis and Antonis A. Argyros
##
## x is any type that implements: norm, sumabs2, maxabs, dot, similar, fill!, copy!, axpy!, broadcast!
## fcur is any type that implements: sumabs2, scale!, similar
## f!(x, out) returns vector f_i(x) in in out
## g!(x, out) returns jacobian in out
##
##############################################################################

function dogleg!(x, fg, fcur, f!::Function, g!::Function; tol =1e-8, maxiter=20)
    const MAX_Δ = 1e16 # minimum trust region radius
    const MIN_Δ = 1e-16 # maximum trust region radius
    const MIN_STEP_QUALITY = 1e-3

    # temporary array
    δgn = similar(x) # gauss newton step
    δsd = similar(x) # steepest descent
    δdiff = similar(x) # δgn - δsd
    δx = similar(x)
    ftrial = similar(fcur)
    ftmp = similar(fcur)

    # temporary arrays used in cgls
    s = similar(x)
    p = similar(x)
    z = similar(x)
    ptmp = similar(x)
    normalization = similar(x)
    q = similar(fcur)
   
    # initialize
    Δ = norm(x)
    f!(x, fcur)
    g!(x, fg)
    scale!(fcur, -1.0)
    residual = sumabs2(fcur)

    iter = 0
    while iter < maxiter 
        iter += 1
        g!(x, fg)
        Ac_mul_B!(δsd, fg, fcur)
        if maxabs(δsd) <= tol
            return iter, true
        end
        A_mul_B!(ftmp, fg, δsd)
        scale!(δsd, sumabs2(δsd)/sumabs2(ftmp))
        gncomputed = false
        ρ = -1.0
        while ρ <= MIN_STEP_QUALITY
            # compute δx
            if norm(δsd) >= Δ
                # Cauchy point is out of the region
                # take largest Cauchy step within the trust region boundary
                scale!(δx, δsd, Δ / norm(δsd))
            else
                if (!gncomputed)
                    fill!(δgn, zero(Float64))
                    cgls!(δgn, fcur, fg, normalization, s, z, p, q, ptmp)
                    # δdiff = δgn - δsd
                    copy!(δdiff, δgn)
                    axpy!(-1.0, δsd,  δdiff)
                    gncomputed = true
                end
                if norm(δgn) <= Δ
                    # Gauss-Newton step is within the region
                    copy!(δx, δgn)
                else
                    # Gauss-Newton step is outside the region
                    # intersection trust region and line Cauchy point and the Gauss-Newton step
                    b = 2 * dot(δsd, δdiff)
                    a = sumabs2(δdiff)
                    c = sumabs2(δsd)
                    tau = (-b + sqrt(b^2 - 4 * a * (c - Δ^2)))/(2*a) 
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
                    residual = trial_residual
                else
                    # unsucessful iteration
                    axpy!(-1.0, δx, x)
                end
                if ρ < 0.25
                   Δ = max(MIN_Δ, Δ / 2)
                elseif ρ > 0.75
                   Δ = min(MAX_Δ, 2 * Δ)
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


##############################################################################
##
## Solve A'AX' = X'b
##
##############################################################################


# r should equal b - Ax0 where x0 is an initial guess for x. It is NOT modified in place
# x, s, p, q are used for storage. s, p should have dimension size(A, 2). q should have simension size(A, 1). 
# Conjugate gradient least square with jacobi normalization

function cgls!(x::Union(AbstractVector{Float64}, Nothing), 
               r::AbstractVector{Float64}, A::AbstractMatrix{Float64}, 
               normalization::AbstractVector{Float64}, s::AbstractVector{Float64}, 
               z::AbstractVector{Float64}, p::AbstractVector{Float64}, 
               q::AbstractVector{Float64}, ptmp; 
               tol::Real=1e-5, maxiter::Int=100)

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




