type MatrixWrapper{TA, Tx}
    A::TA
    d::Tx
end

type VectorWrapper{Ty, Tx}
    y::Ty
    x::Tx
end

function copy!{Tx, Ty}(a::VectorWrapper{Tx, Ty}, b::VectorWrapper{Tx, Ty})
    copy!(a.y, b.y)
    copy!(a.x, b.x)
    return a
end

function fill!(a::VectorWrapper, x)
    fill!(a.y, x)
    fill!(a.x, x)
    return a
end

function scale!(a::VectorWrapper, x)
    scale!(a.y, x)
    scale!(a.x, x)
    return a
end

function axpy!{Tx, Ty}(α, a::VectorWrapper{Tx, Ty}, b::VectorWrapper{Tx, Ty})
    axpy!(α, a.y, b.y)
    axpy!(α, a.x, b.x)
    return b
end

function norm(a::VectorWrapper)
    return sqrt(norm(a.y)^2 + norm(a.x)^2)
end

function A_mul_B!{TA, Tx, Ty}(α::Float64, mw::MatrixWrapper{TA, Tx}, a::Tx, 
                β::Float64, b::VectorWrapper{Ty, Tx})
    A_mul_B!(α, mw.A, a, β, b.y)
    map!((z, x, y)-> β * z + α * x * y, b.x, b.x, a, mw.d)
    return b
end

function Ac_mul_B!{TA, Tx, Ty}(α::Float64, mw::MatrixWrapper{TA, Tx}, a::VectorWrapper{Ty, Tx}, 
                β::Float64, b::Tx)
    Ac_mul_B!(α, mw.A, a.y, β, b)
    map!((z, x, y)-> z + α * x * y, b, b, a.x, mw.d)
    return b
end

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
    normalization = similar(x)

    ftrial = similar(fcur)
    ftmp = similar(fcur)
    zerosvector = similar(x)
    fill!(zerosvector, zero(Float64))

    # for lsmr
    v = similar(x)
    h = similar(x)
    hbar = similar(x)
    u = VectorWrapper(similar(fcur), similar(x))
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
        # dtd = λ diag(J'J)
        sumabs2!(dtd, fg)
        scale!(dtd, λ)
        clamp!(dtd, MIN_DIAGONAL, Inf)
        # solve (J'J + diagm(dtd)) = -J'f
        # we use LSMR with the matrix A = |J         |
        #                                 |diag(dtd) |
        # (recommanded by Michael Saunders)
        fill!(δx, zero(Float64))
        y = VectorWrapper(fcur, zerosvector)
        A = MatrixWrapper(fg, map!(x-> sqrt(x), dtd, dtd))
        cglsiter, conv = lsmr!(δx, y, A, u, v, h, hbar; 
           atol = 1e-8, btol = 1e-8, conlim = 1e8, λ = 0.)
        iter += cglsiter
        # predicted residual
        copy!(ftmp, fcur)
        A_mul_B!(1.0, fg, δx, -1.0, ftmp)
        predicted_residual = sumabs2(ftmp)
        # trial residual
        axpy!(1.0, δx, x)
        f!(x, ftrial)
        trial_residual = sumabs2(ftrial)
        ρ = (residual - trial_residual) / (residual - predicted_residual)
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




##############################################################################
## LSMR
##
## Minimize ||Ax-b||^2 + λ^2 ||x||^2
##
## Arguments:
## x is initial x0. Will equal the solution.
## r is initial b - Ax0
## u are storage arrays of length size(A, 1) = y
## v, h, hbar are storage arrays of length size(A, 2) = x
## 
## Adapted from the BSD-licensed Matlab implementation at
##  http://web.stanford.edu/group/SOL/software/lsmr/
##
## A is anything such that
## A_mul_B!(α, A, b, β, c) updates c -> α Ab + βc
## Ac_mul_B!(α, A, b, β, c) updates c -> α A'b + βc
##############################################################################

function lsmr!(x, r, A, u, v, h, hbar; 
    atol::Real = 1e-8, btol::Real = 1e-8, conlim::Real = 1e8, 
    maxiter::Integer = 100, λ::Real = zero(Float64))

    conlim > 0.0 ? ctol = 1 / conlim : ctol = zero(Float64)

    # form the first vectors u and v (satisfy  β*u = b,  α*v = A'u)
    copy!(u, r)
    β = norm(u)
    β > 0 && scale!(u, 1/β)
    Ac_mul_B!(1.0, A, u, 0.0, v)
    α = norm(v)
    α > 0 && scale!(v, 1/α)

    # Initialize variables for 1st iteration.
    ζbar = α * β
    αbar = α
    ρ = one(Float64)
    ρbar = one(Float64)
    cbar = one(Float64)
    sbar = zero(Float64)

    copy!(h, v)
    fill!(hbar, zero(Float64))

    # Initialize variables for estimation of ||r||.
    βdd = β
    βd = zero(Float64)
    ρdold = one(Float64)
    τtildeold = zero(Float64)
    θtilde  = zero(Float64)
    ζ = zero(Float64)
    d = zero(Float64)

    # Initialize variables for estimation of ||A|| and cond(A).
    normA2 = α^2
    maxrbar = zero(Float64)
    minrbar = 1e100

    # Items for use in stopping rules.
    normb = β
    istop = 7
    normr = β

    # Exit if b = 0 or A'b = zero(Float64).
    normAr = α * β
    if normAr == zero(Float64) 
        return 1, true
    end

    iter = 0
    while iter < maxiter
        iter += 1
        A_mul_B!(1.0, A, v, -α, u)
        β = norm(u)
        if β > 0
            scale!(u, 1/β)
            Ac_mul_B!(1.0, A, u, -β, v)
            α = norm(v)
            α > 0 && scale!(v, 1/α)
        end

        # Construct rotation Qhat_{k,2k+1}.
        αhat = sqrt(αbar^2 + λ^2)
        chat = αbar / αhat
        shat = λ / αhat

        # Use a plane rotation (Q_i) to turn B_i to R_i.
        ρold = ρ
        ρ = sqrt(αhat^2 + β^2)
        c = αhat / ρ
        s = β / ρ
        θnew = s * α
        αbar = c * α

        # Use a plane rotation (Qbar_i) to turn R_i^T to R_i^bar.
        ρbarold = ρbar
        ζold = ζ
        θbar = sbar * ρ
        ρtemp = cbar * ρ
        ρbar = sqrt(cbar^2 * ρ^2 + θnew^2)
        cbar = cbar * ρ / ρbar
        sbar = θnew / ρbar
        ζ = cbar * ζbar
        ζbar = - sbar * ζbar

        # Update h, h_hat, x.
        scale!(hbar, - θbar * ρ / (ρold * ρbarold))
        axpy!(1.0, h, hbar)
        axpy!(ζ / (ρ * ρbar), hbar, x)
        scale!(h, - θnew / ρ)
        axpy!(1.0, v, h)

        ##############################################################################
        ##
        ## Estimate of ||r||
        ##
        ##############################################################################

        # Apply rotation Qhat_{k,2k+1}.
        βacute = chat * βdd
        βcheck = - shat * βdd

        # Apply rotation Q_{k,k+1}.
        βhat = c * βacute
        βdd = - s * βacute
          
        # Apply rotation Qtilde_{k-1}.
        θtildeold = θtilde
        ρtildeold = sqrt(ρdold^2 + θbar^2)
        ctildeold = ρdold / ρtildeold
        stildeold = θbar / ρtildeold
        θtilde = stildeold * ρbar
        ρdold = ctildeold * ρbar
        βd = - stildeold * βd + ctildeold * βhat

        τtildeold = (ζold - θtildeold * τtildeold) / ρtildeold
        τd = (ζ - θtilde * τtildeold) / ρdold
        d  = d + βcheck^2
        normr = sqrt(d + (βd - τd)^2 + βdd^2)

        # Estimate ||A||.
        normA2 = normA2 + β^2
        normA  = sqrt(normA2)
        normA2 = normA2 + α^2

        # Estimate cond(A).
        maxrbar = max(maxrbar, ρbarold)
        if iter > 1 
            minrbar = min(minrbar, ρbarold)
        end
        condA = max(maxrbar, ρtemp) / min(minrbar, ρtemp)
        ##############################################################################
        ##
        ## Test for convergence
        ##
        ##############################################################################

        # Compute norms for convergence testing.
        normAr  = abs(ζbar)
        normx = norm(x)

        # Now use these norms to estimate certain other quantities,
        # some of which will be small near a solution.
        test1 = normr / normb
        test2 = normAr / (normA * normr)
        test3 = 1 / condA
        t1 = test1 / (1 + normA * normx / normb)
        rtol = btol + atol * normA * normx / normb

        # The following tests guard against extremely small values of
        # atol, btol or ctol.  (The user may have set any or all of
        # the parameters atol, btol, conlim  to 0.)
        # The effect is equivalent to the normAl tests using
        # atol = eps,  btol = eps,  conlim = one(Float64)/eps.
        if 1 + test3 <= one(Float64) istop = 6; break end
        if 1 + test2 <= one(Float64) istop = 5; break end
        if 1 + t1 <= one(Float64) istop = 4; break end

        # Allow for tolerances set by the user.
        if test3 <= ctol istop = 3; break end
        if test2 <= atol istop = 2; break end
        if test1 <= rtol  istop = 1; break end
    end
    return iter, (istop != 7) && (istop != 3)
end
    



