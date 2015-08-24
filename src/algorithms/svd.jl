##############################################################################
##
## _copy! alternates between long and wide representation
##
##############################################################################

function _copy!{Tid, Ttime}(ymatrix::Matrix{Float64}, yvector::Vector{Float64}, idsrefs::Vector{Tid}, timesrefs::Vector{Ttime})
    @inbounds @simd for i in 1:length(yvector)
        ymatrix[idsrefs[i], timesrefs[i]] = yvector[i]
    end
end

function _copy!{Tid, Ttime}(yvector::Vector{Float64}, ymatrix::Matrix{Float64},  idsrefs::Vector{Tid}, timesrefs::Vector{Ttime})
    @inbounds @simd for i in 1:length(yvector)
        yvector[i] = ymatrix[idsrefs[i], timesrefs[i]]
    end
end

##############################################################################
##
## Estimate factor model by SVD Method
##
##############################################################################

function fit!{Rid, Rtime}(::Type{Val{:svd}},
                          y::Vector{Float64}, 
                          idf::PooledFactor{Rid}, 
                          timef::PooledFactor{Rtime},
                          ::Ones; 
                          maxiter::Integer = 1e7, 
                          tol::Real = 1e-9,
                          lambda::Real = 0.0)
    lambda == 0.0 || error("The svd method only works with lambda = 0.0")
    length(unique(zip(idf.refs, timef.refs))) == length(y) || error("The svd method only works with unique observation per (id, time)")
    N = size(idf.pool, 1)
    T = size(timef.pool, 1)
    rank = size(idf.pool, 2)
    # initialize at zero for missing values
    res_matrix = A_mul_Bt(idf.pool, timef.pool)
    predict_matrix = deepcopy(res_matrix)
    factors = timef.pool
    variance = Array(Float64, (T, T))
    converged = false
    iterations = maxiter
    f_x = zero(Float64)
    oldf_x = zero(Float64)


    # starts the loop
    iter = 0
    while iter < maxiter
        iter += 1
        (predict_matrix, res_matrix) = (res_matrix, predict_matrix)
        (f_x, oldf_x) = (oldf_x, f_x)
        # transform vector into matrix
        _copy!(res_matrix, y, idf.refs, timef.refs)

        # principal components
        At_mul_B!(variance, res_matrix, res_matrix)
        F = eigfact!(Symmetric(variance), (T - rank + 1):T)
        factors = F[:vectors]
        
        # predict matrix
        A_mul_Bt!(variance, factors, factors)
        A_mul_B!(predict_matrix, res_matrix, variance)

        # check convergence
        f_x = sqeuclidean(vec(predict_matrix), vec(res_matrix))
        if f_x == zero(Float64) || abs(f_x - oldf_x)/f_x < tol 
            converged = true
            iterations = iter
            break
        end
    end
    timef.pool = reverse(factors)
    idf.pool = res_matrix * timef.pool

    return ([iterations], [converged])

end


##############################################################################
##
## Estimate ols models with interactive fixed effects by SVD method
##
##############################################################################


function fit!{Rid, Rtime}(::Type{Val{:svd}},
                          X::Matrix{Float64},
                          M::Matrix{Float64},
                          b::Vector{Float64},
                          y::Vector{Float64},
                          idf::PooledFactor{Rid},
                          timef::PooledFactor{Rtime},
                          ::Ones;
                          maxiter::Integer = 100_000,
                          tol::Real = 1e-9,
                          lambda::Real = 0.0)
    lambda == 0.0 || error("The svd method only works with lambda = 0.0")
    length(unique(zip(idf.refs, timef.refs))) == length(y) || error("The svd method only works with unique observation per (id, time)")
    b = M * y
    new_b = deepcopy(b)
    res = Array(Float64, length(y))
    N = size(idf.pool, 1)
    T = size(timef.pool, 1)
    rank = size(idf.pool, 2)

    # initialize at zero for missing values
    res_matrix = fill(zero(Float64), (N, T))
    predict_matrix = deepcopy(res_matrix)
    factors = Array(Float64, (T, rank))
    variance = Array(Float64, (T, T))
    converged = false
    iterations = maxiter
    f_x = Inf

    # starts the loop
    iter = 0
    while iter < maxiter
        iter += 1
        (new_b, b) = (b, new_b)
        (predict_matrix, res_matrix) = (res_matrix, predict_matrix)

        # Given beta, compute the factors
        copy!(res, y)
        subtract_b!(res, b, X)
        # transform vector into matrix 
        _copy!(res_matrix, res, idf.refs, timef.refs)
        # svd of res_matrix
        At_mul_B!(variance, res_matrix, res_matrix)
        F = eigfact!(Symmetric(variance), (T - rank + 1):T)
        factors = F[:vectors]

        # Given the factors, compute beta
        A_mul_Bt!(variance, factors, factors)
        A_mul_B!(predict_matrix, res_matrix, variance)
        _copy!(res, predict_matrix, idf.refs, timef.refs)
        BLAS.axpy!(-1.0, y, res)
        new_b = - M * res
        # Check convergence
        f_x = chebyshev(b, new_b)
        if f_x < tol 
            converged = true
            iterations = iter
            break
        end
    end

    timef.pool = reverse(factors)
    idf.pool = predict_matrix * timef.pool
    return (b, [iterations], [converged])
end


