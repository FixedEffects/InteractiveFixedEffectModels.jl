##############################################################################
##
## dogleg method
##
## Reference: Is Levenberg-Marquardt the Most Efficient Optimization Algorithm for Implementing Bundle Adjustment? 
## Manolis I.A. Lourakis and Antonis A. Argyros
##
## x is any type that implements: norm, sumabs2, dot, similar, fill!, copy!, axpy!, map!
## fcur is any type that implements: sumabs2(fcur), scale!(fcur, α), similar(fcur), axpy!
## J is a matrix, a SparseMatrixSC, or anything that implements
## sumabs21(vec, J) : updates vec to sumabs(J, 1)
## A_mul_B!(α, A, b, β, c) updates c -> α Ab + βc
## Ac_mul_B!(α, A, b, β, c) updates c -> α A'b + βc
##
## f!(x, out) returns vector f_i(x) in in out
## g!(x, out) returns jacobian in out
##
## x is the initial solution and is transformed in place to the solution
##
##############################################################################
const MIN_Δ = 1e-16 # maximum trust region radius
const MAX_Δ = 1e16 # minimum trust region radius
const MIN_STEP_QUALITY = 1e-3
const GOOD_STEP_QUALITY = 0.75

function dogleg!(x, fcur, f!::Function, J, g!::Function; 
                tol = 1e-8, maxiter = 100, Δ = 1.0)
 
    # temporary array
    δgn = similar(x) # gauss newton step
    δsd = similar(x) # steepest descent
    δdiff = similar(x) # δgn - δsd
    δx = similar(x)
    ftrial = similar(fcur)
    ftmp = similar(fcur)

    # temporary arrays used in computing least square
    alloc = dl_lssolver_alloc(x, fcur, J)

    # initialize
    f!(x, fcur)
    mfcur = scale!(fcur, -1.0)
    residual = sumabs2(fcur)
    iter = 0
    while iter < maxiter 
        iter += 1
        g!(x, J)
        Ac_mul_B!(δsd, J, mfcur)
        A_mul_B!(ftmp, J, δsd)
        scale!(δsd, sumabs2(δsd) / sumabs2(ftmp))
        gncomputed = false
        ρ = -1.0
        while ρ < MIN_STEP_QUALITY
            # compute δx
            if norm(δsd) >= Δ
                # Cauchy point is out of the region
                # take largest Cauchy step within the trust region boundary
                scale!(δx, δsd, Δ / norm(δsd))
            else
                if (!gncomputed)
                    fill!(δgn, zero(Float64))
                    cgls_iter = dl_lssolver!(δgn, mfcur, J, alloc)
                    iter += cgls_iter
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
            axpy!(1.0, δx, x)
            f!(x, ftrial)
            trial_residual = sumabs2(ftrial)

            if abs(residual - trial_residual) <= max(tol^2 * residual, eps()^2)
                return iter, true
            end

            A_mul_B!(ftmp, J, δx)
            axpy!(-1.0, mfcur, ftmp)
            predicted_residual = sumabs2(ftmp)
            ρ = (trial_residual - residual) / (predicted_residual - residual)

            if ρ >= MIN_STEP_QUALITY
                # Successful iteration
                copy!(fcur, ftrial)
                mfcur = scale!(fcur, -1.0)
                g!(x, J)
                residual = trial_residual
            else
                # unsucessful iteration
                axpy!(-1.0, δx, x)
            end
            if ρ < 0.25
               Δ = max(MIN_Δ, Δ / 2)
            elseif ρ > GOOD_STEP_QUALITY
               Δ = min(MAX_Δ, 2 * Δ)
           end          
        end
    end
    return maxiter, false
end

##############################################################################
## 
## Case of Dense Matrix
##
##############################################################################

function dl_lssolver_alloc{T}(x::Vector{T}, mfcur::Vector{T}, J::Matrix{T})
    nothing
end

function dl_lssolver!{T}(δx::Vector{T}, mfcur::Vector{T}, J::Matrix{T}, alloc::Void)
    δx[:] = J \ mfcur
    return 1
end

##############################################################################
## 
## Case where J'J is costly to store: Sparse Matrix, or anything
## that defines two functions
## A_mul_B(α, A, a, β b) that updates b as α A a + β b 
## Ac_mul_B(α, A, a, β b) that updates b as α A' a + β b 
## sumabs21(x, A) that updates x with sumabs2(A)
##
## we use LSMR for the problem J'J \ J' fcur 
## with 1/sqrt(diag(J'J)) as preconditioner
##############################################################################

type MatrixWrapperDogleg{TA, Tx}
    A::TA
    normalization::Tx 
    tmp::Tx
end

function A_mul_B!{TA, Tx}(α::Float64, mw::MatrixWrapperDogleg{TA, Tx}, a::Tx, 
                β::Float64, b)
    map!((x, z) -> x * z, mw.tmp, a, mw.normalization)
    A_mul_B!(α, mw.A, mw.tmp, β, b)
    return b
end

function Ac_mul_B!{TA, Tx}(α::Float64, mw::MatrixWrapperDogleg{TA, Tx}, a, 
                β::Float64, b::Tx)
    Ac_mul_B!(α, mw.A, a, 0.0, mw.tmp)
    map!((x, z) -> x * z, mw.tmp, mw.tmp, mw.normalization)
    axpy!(β, b, mw.tmp)
    copy!(b, mw.tmp)
    return b
end

function dl_lssolver_alloc(x, fcur, J)
    normalization = similar(x)
    tmp = similar(x)
    fill!(tmp, zero(Float64))
    u = similar(fcur)
    v = similar(x)
    h = similar(x)
    hbar = similar(x)
    return normalization, tmp, u, v, h, hbar
end

function dl_lssolver!(δx, mfcur, J, alloc)
    normalization, tmp, u, v, h, hbar = alloc
    sumabs21!(normalization, J)
    map!(x -> x == 0. ? 0. : 1 / sqrt(x), normalization, normalization)
    A = MatrixWrapperDogleg(J, normalization, tmp)
    iter = lsmr!(δx, mfcur, A, u, v, h, hbar)
    map!((x, z) -> x * z, δx, δx, normalization)
    return iter
end

## Particular case of Sparse matrix
function sumabs21!(v::Vector, A::Base.SparseMatrix.SparseMatrixCSC)
    for i in 1:length(v)
        v[i] = sumabs2(sub(nonzeros(A), nzrange(A, i)))
    end
end
