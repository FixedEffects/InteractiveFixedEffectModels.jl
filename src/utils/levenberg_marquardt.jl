##############################################################################
##
## An Inexact Levenberg Marquardt Method for Large Sparse Nonlinear Least Squares
## SJ Wright and J.N. Holt
## 
##############################################################################

function levenberg_marquardt!(x, fg, fcur, f!, g!; tol =1e-8, maxiter=1000, λ=1.0)
    const MAX_λ = 1e16 # minimum trust region radius
    const MIN_λ = 1e-16 # maximum trust region radius
    const MIN_STEP_QUALITY = 1e-3
    const GOOD_STEP_QUALITY = 0.75
    const MIN_DIAGONAL = 1e-6

    converged = false
    iterations = maxiter
    need_jacobian = true

    δx = similar(x)
    dtd = similar(x)
    dtd = similar(x)
    ftrial = similar(fcur)
    ftmp = similar(fcur)

    # for cgls
    s = similar(x)
    p = similar(x)
    z = similar(x)
    ptmp = similar(x)
    ptmp2 = similar(x)
    normalization = similar(x)
    q = similar(fcur)

    # initialize
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
        # dtd = λ^2 diag(J'J)
        sumabs2!(dtd, fg)
        scale!(dtd, λ^2)
        clamp!(dtd, MIN_DIAGONAL, Inf)
        # solve (J'J + diagm(dtd)) = -J'f
        fill!(δx, zero(Float64))
        cglsiter, conv = cgls!(δx, fcur, fg, dtd, normalization, s, z, p, q, ptmp, ptmp2; tol = 0.5)
        iter += cglsiter
        # predicted residual
        A_mul_B!(ftmp, fg, δx)
        predicted_residual = _sumabs2diff(fcur, ftmp)
        # trial residual
        axpy!(1.0, δx, x)
        f!(x, ftrial)
        trial_residual = sumabs2(ftrial)
        ρ = (residual - trial_residual) / (residual - predicted_residual - λ^2 * sumabs2(δx))
        if ρ > GOOD_STEP_QUALITY
            copy!(fcur, ftrial)
            scale!(fcur, -1.0)
            residual = trial_residual
            # increase trust region radius
            if ρ > 0.75
                λ = max(0.1*λ, MIN_λ)
            end
            need_jacobian = true
        else
            # revert update
            axpy!(-1.0, δx, x)
            λ = min(10*λ, MAX_λ)
        end
        if maxabs(δx) < tol
            iterations = iter
            converged = true
            break
        end
    end
    return iterations, converged
end


function _sumabs2diff(x1, x2)
    out = zero(Float64)
    @inbounds @simd for i in 1:length(x1)
        out += abs2(x1[i] - x2[i])
    end
    return out
end


##############################################################################
##
## Solve (A'A + diagm(d))x = A'b by cgls with jacobi preconditioner
## x is the initial guess for x. It is modified in place
## r equals b - Ax0 where x0 is the initial guess for x. It is not modified in place
## s, p, z, ptmp, ptmp2 are used for storage. They have dimension size(A, 2). 
## q is used for storage. It has dimension size(A, 1). 
##
## Stopping rule : Least-squares problems, normal equations, and stopping criteria for the conjugate gradient method Mario Arioli and Serge Gratton 
##
##############################################################################

function cgls!(x, r, A, d, normalization, s, z, p, q, ptmp, ptmp2; 
               tol::Real=1e-5, maxiter::Int=100)

    # Initialization.
    converged = false
    iterations = maxiter 

    Ac_mul_B!(s, A, r)
    sumabs2!(normalization, A)
    axpy!(1.0, d, normalization)
    broadcast!(/, z, s, normalization)
    copy!(p, z)
    ssr0 = dot(s, z)
    ssrold = ssr0  
    ν = sumabs2(r)
    ψ = Float64[]

    iter = 0
    while iter < maxiter
        iter += 1
        A_mul_B!(q, A, p) 
        Ac_mul_B!(ptmp, A, q)
        broadcast!(*, ptmp2, d, p)
        axpy!(1.0, ptmp2, ptmp)
        α = ssrold / dot(ptmp, p)
        push!(ψ, α * ssrold)
        ν -= α * ssrold
        axpy!(α, p, x) 
        axpy!(-α, ptmp, s)
        broadcast!(/, z, s, normalization)
        ssr = dot(s, z)
        if ssr <= 1e-32 || ψ[end] <= 1e-32 || ((iter >= 3) && (sum(sub(ψ, (iter-2):iter)) <= max(eps(), tol^2 * ν)))
            iterations = iter
            converged = true
            break
        end
        β = ssr / ssrold
        # p = s + β p
        scale!(p, β)
        axpy!(1.0, z, p) 
        ssrold = ssr
    end
    return iterations, converged
end




