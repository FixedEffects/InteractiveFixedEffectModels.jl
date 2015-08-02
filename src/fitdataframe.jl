
##############################################################################
#
## Starts from a dataframe and returns a dataframe
##
##############################################################################

function fit(m::SparseFactorModel, 
             f::Formula, 
             df::AbstractDataFrame, 
             vcov_method::AbstractVcovMethod = VcovSimple(); 
             method::Symbol = :ar, 
             lambda::Real = 0.0, 
             subset::Union(AbstractVector{Bool}, Nothing) = nothing, 
             weight::Union(Symbol, Nothing) = nothing, 
             maxiter::Integer = 10000, 
             tol::Real = 1e-20, 
             save = true)

    ##############################################################################
    ##
    ## Transform DataFrame into a set of Matrix and Vectors
    ##
    ##############################################################################

    # initial check
    if method == :svd
        weight == nothing || error("The svd methods does not work with weights")
        lambda == 0.0 || error("The svd method only works with lambda = 0.0")
    elseif method == :ar
        lambda == 0.0 || error("The Gauss-Seidel method only works with lambda = 0.0")
    end

    rf = deepcopy(f)
    ## decompose formula into normal  vs absorbpart
    (has_absorb, absorb_formula, absorb_terms, has_iv, iv_formula, iv_terms, endo_formula, endo_terms) = decompose!(rf)
    if has_iv
        error("partial_out does not support instrumental variables")
    end
    rt = Terms(rf)
    has_regressors = allvars(rf.rhs) != [] || (rt.intercept == true && !has_absorb)

    ## create a dataframe without missing values & negative weights
    vars = allvars(rf)
    vcov_vars = allvars(vcov_method)
    absorb_vars = allvars(absorb_formula)
    factor_vars = [m.id, m.time]
    all_vars = vcat(vars, absorb_vars, factor_vars, vcov_vars)
    all_vars = unique(convert(Vector{Symbol}, all_vars))
    esample = complete_cases(df[all_vars])
    if weight != nothing
        esample &= isnaorneg(df[weight])
        all_vars = unique(vcat(all_vars, weight))
    end
    if subset != nothing
        if length(subset) != size(df, 1)
            error("df has $(size(df, 1)) rows but the subset vector has $(length(subset)) elements")
        end
        esample &= convert(BitArray, subset)
    end
    subdf = df[esample, all_vars]
    main_vars = unique(convert(Vector{Symbol}, vcat(vars, factor_vars)))
    for v in main_vars
        dropUnusedLevels!(subdf[v])
    end

    vcov_method_data = VcovMethodData(vcov_method, subdf)


    # Compute weight
    sqrtw = get_weight(subdf, weight)

    ## Compute factors, an array of AbtractFixedEffects
    if has_absorb
        fes = AbstractFixedEffect[FixedEffect(subdf, a, sqrtw) for a in absorb_terms.terms]
        # in case some FixedEffect is aFixedEffectIntercept, remove the intercept
        if any([typeof(f) <: FixedEffectIntercept for f in fes]) 
            rt.intercept = false
        end
    else
        fes = nothing
    end

    # initialize iterations and converged
    iterations = Int[]
    converged = Bool[]

    # get two dimensions
    id = subdf[m.id]
    time = subdf[m.time]

    ## Compute demeaned X
    mf = simpleModelFrame(subdf, rt, esample)
    if has_regressors
        coef_names = coefnames(mf)
        X = ModelMatrix(mf).m
        broadcast!(*, X, X, sqrtw)
        demean!(X, iterations, converged, fes)
    end

    ## Compute demeaned y
    py = model_response(mf)[:]
    yname = rt.eterms[1]
    if eltype(py) != Float64
        y = convert(py, Float64)
    else
        y = py
    end
    broadcast!(*, y, y, sqrtw)
    oldy = deepcopy(y)
    demean!(y, iterations, converged, fes)

    ##############################################################################
    ##
    ## Estimate Model
    ##
    ##############################################################################

    # initialize factor models at 0.1
    idf = PooledFactor(id.refs, length(id.pool), m.rank)
    timef = PooledFactor(time.refs, length(time.pool), m.rank)
    ## Case simple factor models
    if !has_regressors
        coef = [0.0]
        if method == :svd
            (iterations, converged) = fit_svd!(y, 
                                               idf, 
                                               timef, 
                                               maxiter = maxiter, 
                                               tol = tol)
        elseif method == :ar
            (iterations, converged) = fit_ar!(y, 
                                              idf, 
                                              timef, 
                                              sqrtw, 
                                              maxiter = maxiter, 
                                              tol = tol)
        elseif method == :gd
            (iterations, converged) = fit_gd!(y, 
                                              idf, 
                                              timef, 
                                              sqrtw, 
                                              maxiter = maxiter, 
                                              tol = tol, 
                                              lambda = lambda)
        elseif method == :sgd
            (iterations, converged) = fit_sgd!(y, 
                                               idf, 
                                               timef, 
                                               sqrtw, 
                                               maxiter = maxiter, 
                                               tol = tol, 
                                               lambda = lambda)
        else
            error("method not found")
        end
    else
        # initial b
        crossx = cholfact!(At_mul_B(X, X))
        coef =  crossx \ At_mul_B(X, y)
        # initial loadings
        fit_gd!(y - X * coef, idf, timef, sqrtw, maxiter = 100, tol = 1e-3)
        # dispatch manually to the right method
        if method == :svd
            M = crossx \ X'
            (coef, iterations, converged) = fit_svd!(X, 
                                                     M, 
                                                     coef, 
                                                     y, 
                                                     idf, 
                                                     timef, 
                                                     maxiter = maxiter, 
                                                     tol = tol) 
        elseif method == :ar
            M = crossx \ X'
            (coef, iterations, converged) = fit_ar!(X, 
                                                     M, 
                                                     coef, 
                                                     y, 
                                                     idf, 
                                                     timef, 
                                                     sqrtw, 
                                                     maxiter = maxiter, 
                                                     tol = tol) 
        elseif method == :gd
            M = crossx \ X'
            (coef, iterations, converged) = fit_gd!(X, 
                                                     M, 
                                                     coef, 
                                                     y, 
                                                     idf, 
                                                     timef, 
                                                     sqrtw, 
                                                     maxiter = maxiter, 
                                                     tol = tol, 
                                                     lambda = lambda) 
        else
            error("method not found")
        end
    end



    ##############################################################################
    ##
    ## Return result
    ##
    ##############################################################################

    # compute sum of squared residuals
    residuals = deepcopy(y)
    if has_regressors
        subtract_b!(residuals, coef, X)
    end
    for r in 1:m.rank
        subtract_factor!(residuals, sqrtw, idf, timef, r)
    end
    broadcast!(/, residuals, residuals, sqrtw)
    ess = sumabs2(residuals)


    # save factors and loadings in a dataframe
    if !save 
        augmentdf = DataFrame()
    else
        augmentdf = DataFrame(idf, timef, esample)

        # save residuals in a dataframe
        if all(esample)
            augmentdf[:residuals] = residuals
        else
            augmentdf[:residuals] =  DataArray(Float64, size(augmentdf, 1))
            augmentdf[esample, :residuals] = residuals
        end

        # save fixed effects in a dataframe
        if has_absorb
            # residual before demeaning
            mf = simpleModelFrame(subdf, rt, esample)
            oldresiduals = model_response(mf)[:]
            if has_regressors
                oldX = ModelMatrix(mf).m
                subtract_b!(oldresiduals, coef, oldX)
            end
            for r in 1:m.rank
                subtract_factor!(oldresiduals, fill(one(Float64), length(residuals)), idf, timef, r)
            end
            b = oldresiduals - residuals
            # get fixed effect
            augmentdf = hcat(augmentdf, getfe(fes, b, esample))
        end
    end



    if !has_regressors
        return SparseFactorResult(esample, augmentdf, ess, iterations, converged)
    else
        # compute errors for beta coefficients 
        ## partial out Y on X over dummy_time x loadio
        newfes = getfactors(y, X, coef, idf, timef, sqrtw)
        ym = deepcopy(y)
        Xm = deepcopy(X)
        iterationsv = Int[]
        convergedv = Bool[]
        demean!(ym, iterationsv, convergedv, newfes)
        demean!(Xm, iterationsv, convergedv, newfes)

        residualsm = ym - Xm * coef
        crossxm = cholfact!(At_mul_B(Xm, Xm))

        ## compute the right degree of freedom
        df_absorb_fe = 0
        if has_absorb 
            df_absorb_fe = 0
            ## poor man adjustement of df for clustedered errors + fe: only if fe name != cluster name
            for fe in fes
                if typeof(vcov_method) == VcovCluster && in(fe.name, vcov_vars)
                    df_absorb_fe += 0
                    else
                    df_absorb_fe += sum(fe.scale .!= zero(Float64))
                end
            end
        end
        df_absorb_factors = 0
        df_absorb_factors = 0
        for fe in newfes
            df_absorb_factors += 
                (typeof(vcov_method) == VcovCluster && in(fe.name, vcov_vars)) ? 
                    0 : sum(fe.scale .!= zero(Float64))
        end
        df_residual = size(X, 1) - size(X, 2) - df_absorb_fe - df_absorb_factors 
        if df_residual < 0
            println("There are more parameters than degrees of freedom")
            println("No degree of freedom adjustment for factor structure")
            df_residual = size(X, 1) - size(X, 2) - df_absorb_fe 
        end

        ## estimate vcov matrix
        vcov_data = VcovData(Xm, crossxm, residualsm, df_residual)
        matrix_vcov = vcov!(vcov_method_data, vcov_data)


        # compute various r2
        nobs = size(subdf, 1)
        ess = sumabs2(residualsm)
        tss = compute_ss(residualsm, ym, rt.intercept, sqrtw)
        r2_within = 1 - ess / tss 

        ess = sumabs2(residuals)
        tss = compute_ss(residuals, oldy, rt.intercept || has_absorb, sqrtw)
        r2 = 1 - ess / tss 
        r2_a = 1 - ess / tss * (nobs - rt.intercept) / df_residual 

        return RegressionFactorResult(coef, 
                               matrix_vcov, 
                               esample, 
                               augmentdf, 
                               coef_names, 
                               yname, 
                               f, 
                               nobs, 
                               df_residual, 
                               r2, 
                               r2_a, 
                               r2_within, 
                               ess, 
                               sum(iterations), 
                               all(converged))
    end
end


# Symbol to formul Symbol ~ 0
function fit(m::SparseFactorModel, 
             variable::Symbol, 
             df::AbstractDataFrame, 
             vcov_method::AbstractVcovMethod = VcovSimple();
             method::Symbol = :ar, 
             lambda::Real = 0.0, 
             subset::Union(AbstractVector{Bool},Nothing) = nothing, 
             weight::Union(Symbol,Nothing) = nothing, 
             maxiter::Integer = 10000, 
             tol::Real = 1e-8, 
             save = true)
    formula = Formula(variable, 0)
    fit(m,
        formula,
        df,
        method = method,
        lambda = lambda,
        subset = subset,
        weight = weight,
        subset = subset,
        maxiter = maxiter,
        tol = tol,
        save = true)
end

##############################################################################
##
## DataFrame from factors loadings
##
##############################################################################

function getfactors{Rid, Rtime}(y::Vector{Float64},
                                X::Matrix{Float64},
                                coef::Vector{Float64},
                                id::PooledFactor{Rid},
                                time::PooledFactor{Rtime},
                                sqrtw::AbstractVector{Float64})

    # partial out Y and X with respect to i.id x factors and i.time x loadings
    newfes = AbstractFixedEffect[]
    ans = Array(Float64, length(y))
    for j in 1:size(id.pool, 2)
        for i in 1:length(y)
            ans[i] = time.pool[time.refs[i], j]
        end
        currentid = FixedEffectSlope(id.refs, size(id.pool, 1), sqrtw, ans[:], :id, :time, :(idxtime))
        push!(newfes, currentid)
        for i in 1:length(y)
            ans[i] = id.pool[id.refs[i], j]
        end
        currenttime = FixedEffectSlope(time.refs, size(time.pool, 1), sqrtw, ans[:], :time, :id, :(timexid))
        push!(newfes, currenttime)
    end
    # obtain the residuals and cross 
    return newfes
end





