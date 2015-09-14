##############################################################################
##
## An Inexact Levenberg Marquardt Method for Large Sparse Nonlinear Least Squares
##
##############################################################################

function levenberg_marquardt!(x, fg, fcur, f!, g!; tol =1e-8, maxiter=1000, λ=1.0)
    const MAX_λ = 1e16 # minimum trust region radius
    const MIN_λ = 1e-16 # maximum trust region radius
    const MIN_STEP_QUALITY = 1e-3
    const MIN_DIAGONAL = 1e-6 # lower bound on values of diagonal matrix used to regularize the trust region step

    converged = false
    iterations = maxiter
    need_jacobian = true

    δx = similar(x)
    dtd = similar(x)
    rhs = similar(x)
    dtd = similar(x)
    ftrial = similar(fcur)
    ftmp = similar(fcur)

    # for cgls
    s = similar(x)
    p = similar(x)
    z = similar(x)
    ptmp = similar(x)
    ptmp2 = similar(ptmp)
    normalization = similar(x)
    q = similar(fcur)

    f!(x, fcur)
    scale!(fcur, -1.0)
    residual = sumabs2(fcur)

    iter = 0
    while iter < maxiter 
        iter += 1
        if need_jacobian
            g!(x, fg)
            need_jacobian = false
        end
        sumabs2!(dtd, fg)
        scale!(dtd, λ^2)
        fill!(δx, zero(Float64))
        cglsiter, conv = cgls!(δx, fcur, fg, dtd, normalization, s, z, p, q, ptmp, ptmp2; maxiter = 10)
        iter += cglsiter
        out = predicted_value!(ftmp, fcur, fg, δx)
        # try to update
        axpy!(1.0, δx, x)
        f!(x, ftrial)
        trial_residual = sumabs2(ftrial)
        ρ = (residual - trial_residual) / (residual - out - λ^2 * sumabs2(δx))
        if ρ < 0.01
            axpy!(-1.0, δx, x)
            λ = min(10*λ, MAX_λ)
        else
            copy!(fcur, ftrial)
            scale!(fcur, -1.0)
            residual = trial_residual
            # increase trust region radius
            if ρ > 0.75
                λ = max(0.1*λ, MIN_λ)
            end
            need_jacobian = true
        end
        if maxabs(δx) < tol
            iterations = iter
            converged = true
            break
        end
    end
    return iterations, converged
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

function cgls!(x, r, A, dtd, normalization, s, z, p, q, ptmp, ptmp2; 
               tol::Real=1e-5, maxiter::Int=100)

    # Initialization.
    converged = false
    iterations = maxiter 

    Ac_mul_B!(s, A, r)
    sumabs2!(normalization, A)
    axpy!(1.0, dtd, normalization)
    broadcast!(/, z, s, normalization)
    copy!(p, z)
    normS0 = dot(s, z)
    normSold = normS0  


    iter = 0
    while iter < maxiter
        iter += 1
        A_mul_B!(q, A, p) 
        Ac_mul_B!(ptmp, A, q)
        broadcast!(*, ptmp2, dtd, p)
        axpy!(1.0, ptmp2, ptmp)
        α = normSold / dot(ptmp, p)
        # x = x + αp
        x == nothing || axpy!(α, p, x) 
        axpy!(-α, ptmp, s)
        broadcast!(/, z, s, normalization)
        normS = dot(s, z)
        if α * maxabs(q) <= tol || normS/normS0 <= tol
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




