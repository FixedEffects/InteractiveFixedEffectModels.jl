
##############################################################################
##
## Fit is the only exported function
##
##############################################################################

function reg(df::AbstractDataFrame, 
             f::Formula, 
             m::InteractiveFixedEffectFormula,
             feformula::FixedEffectFormula, 
             vcovformula::AbstractVcovFormula,
             weightformula::WeightFormula; 
             method::Symbol = :dogleg, 
             lambda::Number = 0.0, 
             subset::Union{AbstractVector{Bool}, Void} = nothing, 
             maxiter::Integer = 10_000, 
             tol::Real = 1e-9, 
             save::Union{Bool, Void} = nothing)

    ##############################################################################
    ##
    ## Transform DataFrame -> Matrix
    ##
    ##############################################################################

    ## parse formula 
    rf = deepcopy(f)
    (has_iv, iv_formula, iv_terms, endo_formula, endo_terms) = decompose_iv!(rf)
    if has_iv
        error("partial_out does not support instrumental variables")
    end
    has_absorb = feformula.arg != nothing
    has_weight = (weightformula.arg != nothing)


    rt = Terms(rf)
    has_regressors = allvars(rf.rhs) != [] || (rt.intercept == true && !has_absorb)
    # change default if has_regressors
    if save == nothing 
        save = !has_regressors
    end
    ## create a dataframe without missing values & negative weights
    vars = allvars(rf)
    vcov_vars = allvars(vcovformula)
    absorb_vars = allvars(feformula)
    factor_vars = [m.id, m.time]
    rem = setdiff(absorb_vars, factor_vars)
    if length(rem) > 0
        error("The categorical variable $(rem[1]) appears in @fe but does not appear in @ife. Simply add it in @formula instead")
    end
    all_vars = vcat(vars, absorb_vars, factor_vars, vcov_vars)
    all_vars = unique(convert(Vector{Symbol}, all_vars))
    esample = completecases(df[all_vars])
    if has_weight
        esample &= isnaorneg(df[weightformula.arg])
        all_vars = unique(vcat(all_vars, weightformula.arg))
    end
    if subset != nothing
        if length(subset) != size(df, 1)
            error("df has $(size(df, 1)) rows but the subset vector has $(length(subset)) elements")
        end
        esample &= subset
    end
    subdf = df[esample, all_vars]
    main_vars = unique(convert(Vector{Symbol}, vcat(vars, factor_vars)))
    for v in main_vars
        dropUnusedLevels!(subdf[v])
    end

    # Compute data needed for errors
    vcov_method_data = VcovMethod(subdf, vcovformula)

    # Compute weight
    sqrtw = get_weight(subdf, weightformula)

    ## Compute factors, an array of AbtractFixedEffects
    if has_absorb
        fes = FixedEffect[FixedEffect(subdf, a, sqrtw) for a in Terms(feformula).terms]
        # in case some FixedEffect is aFixedEffectIntercept, remove the intercept
        if any([typeof(f.interaction) <: Ones for f in fes]) 
            rt.intercept = false
        end
        pfe = FixedEffectProblem(fes, Val{:lsmr})
    else
        pfe = nothing
    end

    iterations = 0
    converged = false
    # get two dimensions
    id = subdf[m.id]
    time = subdf[m.time]

    ##############################################################################
    ##
    ## Construict vector y and matrix X
    ##
    ##############################################################################

    mf = ModelFrame2(rt, subdf, esample)

    # Compute demeaned X
    if has_regressors
        coef_names = coefnames(mf)
        X = ModelMatrix(mf).m
        broadcast!(*, X, X, sqrtw)
        residualize!(X, pfe, Int[], Bool[])
    end

    # Compute demeaned y
    py = model_response(mf)[:]
    yname = rt.eterms[1]
    if eltype(py) != Float64
        y = convert(py, Float64)
    else
        y = py
    end
    broadcast!(*, y, y, sqrtw)
    oldy = deepcopy(y)
    residualize!(y, pfe, Int[], Bool[])

    ##############################################################################
    ##
    ## Estimate Model on Matrix
    ##
    ##############################################################################

    # initialize factor models at 0.1
    idpool = fill(0.1, length(id.pool), m.rank)
    timepool = fill(0.1, length(time.pool), m.rank)
  
    if !has_regressors
        fp = FactorModel(y, sqrtw, id.refs, time.refs, m.rank)
        fs = FactorSolution(idpool, timepool)
        # factor model 

        (fs, iterations, converged) = 
            fit!(Val{method}, fp, fs; maxiter = maxiter, tol = tol, lambda = lambda)
    else 
        # interactive fixed effect
        # initialize fs
        coef = X \ y
        fp = FactorModel(y - X * coef, sqrtw, id.refs, time.refs, m.rank)
        fs = FactorSolution(idpool, timepool)
        fit!(Val{:levenberg_marquardt}, fp, fs; maxiter = 100, tol = 1e-3, lambda = lambda)

        fs = InteractiveFixedEffectsSolution(coef, fs.idpool, fs.timepool)
        fp = InteractiveFixedEffectsModel(y, sqrtw, X, id.refs, time.refs, m.rank)
        ym = deepcopy(y)
        Xm = deepcopy(X)

        while true 
            # estimate the model
           (fs, iterations, converged) = 
                fit!(Val{method}, fp, fs; maxiter = maxiter, tol = tol, lambda = lambda)
            # check that I obtain the same coefficient if I solve
            # y ~ x + γ1 x factors + γ2 x loadings
            # if not, this means fit! ended up on a a local minimum. 
            # restart with randomized coefficients, factors, loadings
            newpfe = FixedEffectProblem(getfactors(fp, fs), Val{:lsmr})
            residualize!(ym, newpfe, Int[], Bool[], tol = tol, maxiter = maxiter)
            residualize!(Xm, newpfe, Int[], Bool[], tol = tol, maxiter = maxiter)
            ydiff = Xm * (fs.b - Xm \ ym)
            if iterations >= maxiter || norm(ydiff)  <= 0.01 * norm(y)
                break
            end
            info("Algorithm ended up on a local minimum. Restarting from a new, random, x0.")
            map!(x -> randn() * x, fs, fs)
            copy!(ym, y)
            copy!(Xm, X)
        end
    end

    ##############################################################################
    ##
    ## Compute residuals
    ##
    ##############################################################################

    # compute residuals
    fp = FactorModel(deepcopy(y), sqrtw, id.refs, time.refs, m.rank)
    if has_regressors
        BLAS.gemm!('N', 'N', -1.0, X, fs.b, 1.0, fp.y)
    end
    subtract_factor!(fp, fs)
    broadcast!(/, fp.y, fp.y, sqrtw)
    residuals = fp.y
    ##############################################################################
    ##
    ## Compute errors
    ##
    ##############################################################################
   
    if !has_regressors
        ess = sumabs2(residuals)
    else
        residualsm = ym .- Xm * fs.b
        crossxm = cholfact!(At_mul_B(Xm, Xm))
        ## compute the right degree of freedom
        df_absorb_fe = 0
        if has_absorb 
            df_absorb_fe = 0
            ## poor man adjustement of df for clustedered errors + fe: only if fe name != cluster name
            for fe in fes
                if typeof(vcovformula) == VcovClusterFormula && in(fe.factorname, vcov_vars)
                    df_absorb_fe += 0
                else
                    df_absorb_fe += sum(fe.scale .!= zero(Float64))
                end
            end
        end
        df_absorb_factors = 0
        newfes = getfactors(fp, fs)
        for fe in newfes
            df_absorb_factors += 
                (typeof(vcovformula) == VcovClusterFormula && in(fe.factorname, vcov_vars)) ? 
                    0 : sum(fe.scale .!= zero(Float64))
        end
        df_residual = max(size(X, 1) - size(X, 2) - df_absorb_fe - df_absorb_factors, 1)

        ## estimate vcov matrix
        vcov_data = VcovData(Xm, crossxm, residualsm, df_residual)
        matrix_vcov = vcov!(vcov_method_data, vcov_data)

        # compute various r2
        nobs = size(subdf, 1)
        ess = sumabs2(residualsm)
        tss = compute_tss(ym, rt.intercept, sqrtw)
        r2_within = 1 - ess / tss 

        ess = sumabs2(residuals)
        tss = compute_tss(oldy, rt.intercept || has_absorb, sqrtw)
        r2 = 1 - ess / tss 
        r2_a = 1 - ess / tss * (nobs - rt.intercept) / df_residual 
    end

    ##############################################################################
    ##
    ## Save factors and loadings in a dataframe
    ##
    ##############################################################################

    if !save 
        augmentdf = DataFrame()
    else

        augmentdf = DataFrame(fp, fs, esample)
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
            mf = ModelFrame2(rt, subdf, esample)
            oldresiduals = model_response(mf)[:]
            if has_regressors
                oldX = ModelMatrix(mf).m
                BLAS.gemm!('N', 'N', -1.0, oldX, coef, 1.0, oldresiduals)
            end
            fp = FactorModel(oldresiduals, sqrtw, id.refs, time.refs, m.rank)
            subtract_factor!(fp, fs)
            axpy!(-1.0, residuals, oldresiduals)
            # get fixed effect
            augmentdf = hcat(augmentdf, getfe!(pfe, oldresiduals, esample))
        end
    end


    if !has_regressors
        return FactorResult(esample, augmentdf, ess, iterations, converged)
    else
        return InteractiveFixedEffectsResult(fs.b, matrix_vcov, esample, augmentdf, 
            coef_names, yname, f, nobs, df_residual, r2, r2_a, r2_within, 
            ess, sum(iterations), all(converged))
    end
end

function reg(df::AbstractDataFrame, f::Formula, ife::InteractiveFixedEffectFormula; kwargs...) 
    reg(df, f, ife, @fe(), @vcov(), @weight(); kwargs...)
end
function reg(df::AbstractDataFrame, f::Formula, ife::InteractiveFixedEffectFormula, feformula::FixedEffectFormula; kwargs...) 
    reg(df, f, ife, feformula, @vcov(), @weight(); kwargs...)
end
function reg(df::AbstractDataFrame, f::Formula, ife::InteractiveFixedEffectFormula, vcovformula::AbstractVcovFormula; kwargs...) 
    reg(df, f, ife, @fe(), vcovformula, @weight(); kwargs...)
end
function reg(df::AbstractDataFrame, f::Formula, ife::InteractiveFixedEffectFormula, weightformula::WeightFormula; kwargs...) 
    reg(df, f, ife, @fe(), @vcov(), weightformula; kwargs...)
end
function reg(df::AbstractDataFrame, f::Formula, ife::InteractiveFixedEffectFormula,  vcovformula::AbstractVcovFormula, weightformula::WeightFormula; kwargs...) 
    reg(df, f, ife, @fe(), vcovformula, weightformula; kwargs...)
end
function reg(df::AbstractDataFrame, f::Formula, ife::InteractiveFixedEffectFormula, feformula::FixedEffectFormula, weightformula::WeightFormula; kwargs...) 
    reg(df, f, ife, feformula, @vcov(), weightformula; kwargs...)
end
function reg(df::AbstractDataFrame, f::Formula, ife::InteractiveFixedEffectFormula, feformula::FixedEffectFormula, vcovformula::AbstractVcovFormula; kwargs...) 
    reg(df, f, ife, feformula, vcovformula, @weight(); kwargs...)
end

