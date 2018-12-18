
##############################################################################
##
## Fit is the only exported function
##
##############################################################################


function regife(df::AbstractDataFrame, m::Model; kwargs...)
    regife(df, m.f; m.dict..., kwargs...)
end

function regife(df::AbstractDataFrame, 
             f::Formula;
             ife::Union{Symbol, Expr, Nothing} = nothing, 
             fe::Union{Symbol, Expr, Nothing} = nothing, 
             vcov::Union{Symbol, Expr, Nothing} = :(simple()), 
             weights::Union{Symbol, Expr, Nothing} = nothing, 
             subset::Union{Symbol, Expr, Nothing} = nothing, 
             method::Symbol = :dogleg, 
             lambda::Number = 0.0, 
             maxiter::Integer = 10_000, 
             tol::Real = 1e-9, 
             save::Union{Bool, Nothing} = nothing)

    ##############################################################################
    ##
    ## Transform DataFrame -> Matrix
    ##
    ##############################################################################
    feformula = fe
    if isa(vcov, Symbol)
        vcovformula = VcovFormula(Val{vcov})
    else 
        vcovformula = VcovFormula(Val{vcov.args[1]}, (vcov.args[i] for i in 2:length(vcov.args))...)
    end
    m = InteractiveFixedEffectFormula(ife)
    ## parse formula 
    rf = deepcopy(f)
    (has_iv, iv_formula, iv_terms, endo_formula, endo_terms) = FixedEffectModels.decompose_iv!(rf)
    if has_iv
        error("partial_out does not support instrumental variables")
    end
    has_absorb = feformula != nothing
    has_weights = (weights != nothing)


    rt = Terms(rf)
    has_regressors = allvars(rf.rhs) != [] || (rt.intercept == true && !has_absorb)
    # change default if has_regressors
    if save == nothing 
        save = !has_regressors
    end
    ## create a dataframe without missing values & negative weightss
    vars = allvars(rf)
    vcov_vars = allvars(vcovformula)
    absorb_vars = allvars(feformula)
    factor_vars = vcat(allvars(m.id), allvars(m.time))
    rem = setdiff(absorb_vars, factor_vars)
    if length(rem) > 0
        error("The categorical variable $(rem[1]) appears in @fe but does not appear in @ife. Simply add it in @formula instead")
    end
    all_vars = vcat(vars, absorb_vars, factor_vars, vcov_vars)
    all_vars = unique(Symbol.(all_vars))
    esample = completecases(df[all_vars])
    if has_weights
        esample .&= isnaorneg(df[weights])
        all_vars = unique(vcat(all_vars, weights))
    end
    if subset != nothing
        subset = eval(evaluate_subset(df, subset))
        if length(subset) != size(df, 1)
            error("df has $(size(df, 1)) rows but the subset vector has $(length(subset)) elements")
        end
        esample .&= subset
    end
    main_vars = unique(Symbol.(vcat(vars, factor_vars)))


    # Compute data needed for errors
    vcov_method_data = VcovMethod(df[esample, unique(Symbol.(vcov_vars))], vcovformula)

     # Compute weights
     sqrtw = get_weights(df, esample, weights)

    ## Compute factors, an array of AbtractFixedEffects
    if has_absorb
        fes, ids = FixedEffectModels.parse_fixedeffect(df, Terms(@eval(@formula(nothing ~ $(feformula)))))
        # in case some FixedEffect is a FixedEffectIntercept, remove the intercept
        if any([isa(x.interaction, Ones) for x in fes]) 
            rt.intercept = false
        end
        fes = FixedEffect[FixedEffectModels._subset(fe, esample) for fe in fes]
        pfe = FixedEffectModels.FixedEffectMatrix(fes, sqrtw, Val{:lsmr})
    else
        pfe = nothing
    end



    iterations = 0
    converged = false
    # get two dimensions

    if isa(m.id, Symbol)
        # always factorize
        id = group(df[esample, m.id])
    else
        factorvars, interactionvars = _split(df, allvars(m.id))
        id = group((df[esample, v] for v in factorvars)...)
    end
    if isa(m.time, Symbol)
        # always factorize
        time = group(df[esample, m.time])
    else
        factorvars, interactionvars = _split(df, allvars(m.time))
        time = group((df[esample, v] for v in factorvars)...)
    end

    ##############################################################################
    ##
    ## Construict vector y and matrix X
    ##
    ##############################################################################

    mf = ModelFrame2(rt, df, esample)

    # Compute demeaned X
    if has_regressors
        coef_names = coefnames(mf)
        X = ModelMatrix(mf).m
        X .= X .* sqrtw 
        if has_absorb
            FixedEffectModels.solve_residuals!(X, pfe)
        end
    end

    # Compute demeaned y
    py = model_response(mf)[:]
    yname = rt.eterms[1]
    if eltype(py) != Float64
        y = convert(py, Float64)
    else
        y = py
    end
    y .= y .* sqrtw 
    oldy = copy(y)
    if has_absorb
        FixedEffectModels.solve_residuals!(y, pfe)
    end

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
        ym = copy(y)
        Xm = copy(X)

        while true 
            # estimate the model
           (fs, iterations, converged) = 
                fit!(Val{method}, fp, fs; maxiter = maxiter, tol = tol, lambda = lambda)
            # check that I obtain the same coefficient if I solve
            # y ~ x + γ1 x factors + γ2 x loadings
            # if not, this means fit! ended up on a a local minimum. 
            # restart with randomized coefficients, factors, loadings
            newpfe = FixedEffectModels.FixedEffectMatrix(getfactors(fp, fs), sqrtw, Val{:lsmr})
            FixedEffectModels.solve_residuals!(ym, newpfe, tol = tol, maxiter = maxiter)
            FixedEffectModels.solve_residuals!(Xm, newpfe, tol = tol, maxiter = maxiter)
            ydiff = Xm * (fs.b - Xm \ ym)
            if iterations >= maxiter || norm(ydiff)  <= 0.01 * norm(y)
                break
            end
            @info "Algorithm ended up on a local minimum. Restarting from a new, random, x0."
            map!(x -> randn() * x, fs, fs)
            copyto!(ym, y)
            copyto!(Xm, X)
        end
    end

    ##############################################################################
    ##
    ## Compute residuals
    ##
    ##############################################################################

    # compute residuals
    fp = FactorModel(copy(y), sqrtw, id.refs, time.refs, m.rank)
    if has_regressors
        gemm!('N', 'N', -1.0, X, fs.b, 1.0, fp.y)
    end
    subtract_factor!(fp, fs)
    fp.y .= fp.y ./ sqrtw
    residuals = fp.y
    ##############################################################################
    ##
    ## Compute errors
    ##
    ##############################################################################
    if !has_regressors
        rss = sum(abs2, residuals)
    else
        residualsm = ym .- Xm * fs.b
        crossxm = cholesky!(Xm' * Xm)
        ## compute the right degree of freedom
        df_absorb_fe = 0
        if has_absorb 
            for fe in fes
                df_absorb_fe += length(unique(fe.refs))
            end
        end
        dof_residual = max(size(X, 1) - size(X, 2) - df_absorb_fe, 1)

        ## estimate vcov matrix
        vcov_data = VcovData(Xm, crossxm, residualsm, dof_residual)
        matrix_vcov = vcov!(vcov_method_data, vcov_data)

        # compute various r2
        nobs = sum(esample)
        rss = sum(abs2, residualsm)
        tss = compute_tss(ym, rt.intercept, sqrtw)
        r2_within = 1 - rss / tss 

        rss = sum(abs2, residuals)
        tss = compute_tss(oldy, rt.intercept || has_absorb, sqrtw)
        r2 = 1 - rss / tss 
        r2_a = 1 - rss / tss * (nobs - rt.intercept) / dof_residual 
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
            augmentdf[:residuals] =  Vector{Union{Float64, Missing}}(missing, size(augmentdf, 1))
            augmentdf[esample, :residuals] = residuals
        end

        # save fixed effects in a dataframe
        if has_absorb
            # residual before demeaning
            mf = ModelFrame2(rt, df, esample)
            oldresiduals = model_response(mf)[:]
            if has_regressors
                oldX = ModelMatrix(mf).m
                gemm!('N', 'N', -1.0, oldX, coef, 1.0, oldresiduals)
            end
            fp = FactorModel(oldresiduals, sqrtw, id.refs, time.refs, m.rank)
            subtract_factor!(fp, fs)
            axpy!(-1.0, residuals, oldresiduals)
            # get fixed effect
            newfes, b, c = FixedEffectModels.solve_coefficients!(oldresiduals, pfe; tol = tol, maxiter = maxiter)
            for j in 1:length(fes)
                augmentdf[ids[j]] = Vector{Union{Float64, Missing}}(missing, length(esample))
                augmentdf[esample, ids[j]] = newfes[j]
            end
        end
    end


    if !has_regressors
        return FactorResult(esample, augmentdf, rss, iterations, converged)
    else
        return InteractiveFixedEffectsResult(fs.b, matrix_vcov, esample, augmentdf, 
            coef_names, yname, f, nobs, dof_residual, r2, r2_a, r2_within, 
            rss, sum(iterations), all(converged))
    end
end



function evaluate_subset(df, ex::Expr)
    if ex.head == :call
        return Expr(ex.head, ex.args[1], (evaluate_subset(df, ex.args[i]) for i in 2:length(ex.args))...)
    else
        return Expr(ex.head, (evaluate_subset(df, ex.args[i]) for i in 1:length(ex.args))...)
    end
end
evaluate_subset(df, ex::Symbol) = df[ex]
evaluate_subset(df, ex)  = ex