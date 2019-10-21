
##############################################################################
##
## Fit is the only exported function
##
##############################################################################
function regife(df, m::FixedEffectModels.ModelTerm; kwargs...)
    regife(df, m.f; m.dict..., kwargs...)
end

function regife(df, f::FormulaTerm, vcov::CovarianceEstimator = Vcov.simple();
            weights::Union{Symbol, Nothing} = nothing, 
            subset::Union{AbstractVector, Nothing} = nothing,
             method::Symbol = :dogleg, 
             lambda::Number = 0.0, 
             maxiter::Integer = 10_000, 
             tol::Real = 1e-9, 
             save::Union{Bool, Nothing} = nothing,
             contrasts::Dict = Dict{Symbol, Any}(),
             feformula::Union{Symbol, Expr, Nothing} = nothing,
            ifeformula::Union{Symbol, Expr, Nothing} = nothing,
            vcovformula::Union{Symbol, Expr, Nothing} = nothing,
            subsetformula::Union{Symbol, Expr, Nothing} = nothing)

    ##############################################################################
    ##
    ## Transform DataFrame -> Matrix
    ##
    ##############################################################################
    df = DataFrame(df; copycols = false)

    # to deprecate
    if vcovformula != nothing
        if (vcovformula == :simple) | (vcovformula == :(simple()))
            vcov = Vcov.Simple()
        elseif (vcovformula == :robust) | (vcovformula == :(robust()))
            vcov = Vcov.Robust()
        else
            vcov = Vcov.cluster(StatsModels.termvars(@eval(@formula(0 ~ $(vcovformula.args[2]))))...)
        end
    end
    if subsetformula != nothing
        subset = eval(evaluate_subset(df, subsetformula))
    end

    if  (ConstantTerm(0) ∉ FixedEffectModels.eachterm(f.rhs)) & (ConstantTerm(1) ∉ FixedEffectModels.eachterm(f.rhs))
        formula = FormulaTerm(f.lhs, tuple(ConstantTerm(1), FixedEffectModels.eachterm(f.rhs)...))
    end

    formula, formula_endo, formula_iv = FixedEffectModels.decompose_iv(f)

    m, formula = parse_interactivefixedeffect(df, formula)
    if ifeformula != nothing # remove after depreciation
        m = OldInteractiveFixedEffectFormula(ifeformula)
    end

    ## parse formula 
    if formula_iv != nothing
        error("partial_out does not support instrumental variables")
    end
    has_weights = (weights != nothing)


    ## create a dataframe without missing values & negative weightss
    vars = StatsModels.termvars(formula)
    if feformula != nothing # remove after depreciation
        vars = vcat(vars, StatsModels.termvars(@eval(@formula(0 ~ $(feformula)))))
    end
    factor_vars = [m.id, m.time]
    all_vars = unique(vcat(vars, factor_vars))
    esample = completecases(df[!, all_vars])
    esample .&= Vcov.completecases(df, vcov)
    if has_weights
        esample .&= BitArray(!ismissing(x) & (x > 0) for x in df[!, weights])
        all_vars = unique(vcat(all_vars, weights))
    end
    if subset != nothing
        if length(subset) != size(df, 1)
            error("df has $(size(df, 1)) rows but the subset vector has $(length(subset)) elements")
        end
        esample .&= BitArray(!ismissing(x) && x for x in subset)
    end
    main_vars = unique(vcat(vars, factor_vars))


    # Compute data needed for errors
    vcov_method_data = Vcov.materialize(view(df, esample,:), vcov)

     # Compute weights
    sqrtw = Ones{Float64}(sum(esample))
    if has_weights
        sqrtw = convert(Vector{Float64}, sqrt.(view(df, esample, weights)))
    end
    for a in FixedEffectModels.eachterm(formula.rhs)
       if has_fe(a)
           isa(a, InteractionTerm) && error("Fixed effects cannot be interacted")
           Symbol(FixedEffectModels.fesymbol(a)) ∉ factor_vars && error("FixedEffect should correspond to id or time dimension of the factor model")
       end
    end
    fes, ids, formula = FixedEffectModels.parse_fixedeffect(df, formula)
    if feformula != nothing # remove after depreciation
        feformula = @eval(@formula(0 ~ $(feformula)))
        fes, ids = FixedEffectModels.oldparse_fixedeffect(df, feformula)
    end 
    has_fes = !isempty(fes)
    has_fes_intercept = false
    ## Compute factors, an array of AbtractFixedEffects
    if has_fes
        if any([isa(fe.interaction, Ones) for fe in fes])
                formula = FormulaTerm(formula.lhs, tuple(ConstantTerm(0), (t for t in FixedEffectModels.eachterm(formula.rhs) if t!= ConstantTerm(1))...))
                has_fes_intercept = true
        end
        fes = FixedEffect[FixedEffectModels._subset(fe, esample) for fe in fes]
        feM = FixedEffectModels.AbstractFixedEffectSolver{Float64}(fes, sqrtw, Val{:lsmr})
    end


    has_intercept = ConstantTerm(1) ∈ FixedEffectModels.eachterm(formula.rhs)


    iterations = 0
    converged = false
    # get two dimensions

    id = FixedEffects.group(df[esample, m.id])
    time = FixedEffects.group(df[esample, m.time])

    ##############################################################################
    ##
    ## Construict vector y and matrix X
    ##
    ##############################################################################
    subdf = columntable(df[esample, unique(vcat(vars))])

    formula_schema = apply_schema(formula, schema(formula, subdf, contrasts), StatisticalModel)

    y = convert(Vector{Float64}, response(formula_schema, subdf))
    oldy = copy(y)
    X = convert(Matrix{Float64}, modelmatrix(formula_schema, subdf))

    # change default if has_regressors
    has_regressors = size(X, 2) > 0
    if save == nothing 
        save = !has_regressors
    end



    # Compute demeaned X
    yname, coef_names = coefnames(formula_schema)
    if !isa(coef_names, Vector)
        coef_names = [coef_names]
    end
    yname = Symbol(yname)
    coef_names = Symbol.(coef_names)



    if has_fes
        FixedEffectModels.solve_residuals!(y, feM)
        FixedEffectModels.solve_residuals!(X, feM)
     end
     y .= y .* sqrtw
     X .= X .* sqrtw

 

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
            newfeM = FixedEffectModels.AbstractFixedEffectSolver{Float64}(getfactors(fp, fs), sqrtw, Val{:lsmr})
            ym .= ym ./sqrtw
            FixedEffectModels.solve_residuals!(ym, newfeM, tol = tol, maxiter = maxiter)
            ym .= ym .* sqrtw

            Xm .= Xm ./sqrtw
            FixedEffectModels.solve_residuals!(Xm, newfeM, tol = tol, maxiter = maxiter)
            Xm .= Xm .* sqrtw
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
        LinearAlgebra.BLAS.gemm!('N', 'N', -1.0, X, fs.b, 1.0, fp.y)
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
        crossxm = cholesky!(Symmetric(Xm' * Xm))
        ## compute the right degree of freedom
        df_absorb_fe = 0
        if has_fes 
            for fe in fes
                df_absorb_fe += length(unique(fe.refs))
            end
        end
        dof_residual = max(size(X, 1) - size(X, 2) - df_absorb_fe, 1)

        ## estimate vcov matrix
        vcov_data = FixedEffectModels.VcovData(Xm, crossxm, residualsm, dof_residual)
        matrix_vcov = StatsBase.vcov(vcov_data, vcov_method_data)
        # compute various r2
        nobs = sum(esample)
        rss = sum(abs2, residualsm)
        _tss = FixedEffectModels.tss(ym, has_intercept || has_fes_intercept, sqrtw)
        r2_within = 1 - rss / _tss 

        rss = sum(abs2, residuals)
        _tss = FixedEffectModels.tss(oldy, has_intercept || has_fes_intercept, sqrtw)
        r2 = 1 - rss / _tss 
        r2_a = 1 - rss / _tss * (nobs - has_intercept) / dof_residual 
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
            augmentdf[!, :residuals] = residuals
        else
            augmentdf[!, :residuals] =  Vector{Union{Float64, Missing}}(missing, size(augmentdf, 1))
            augmentdf[esample, :residuals] = residuals
        end

        # save fixed effects in a dataframe
        if has_fes
            # residual before demeaning
             oldresiduals = convert(Vector{Float64}, response(formula_schema, subdf))
             oldresiduals .= oldresiduals .* sqrtw
             oldX = convert(Matrix{Float64}, modelmatrix(formula_schema, subdf))
             oldX .= oldX .* sqrtw
            if has_regressors
                LinearAlgebra.BLAS.gemm!('N', 'N', -1.0, oldX, coef, 1.0, oldresiduals)
            end
            fp = FactorModel(oldresiduals, sqrtw, id.refs, time.refs, m.rank)
            subtract_factor!(fp, fs)
            axpy!(-1.0, residuals, oldresiduals)
            # get fixed effect
            newfes, b, c = FixedEffectModels.solve_coefficients!(oldresiduals, feM; tol = tol, maxiter = maxiter)
            for j in 1:length(fes)
                augmentdf[!, ids[j]] = Vector{Union{Float64, Missing}}(missing, length(esample))
                augmentdf[esample, ids[j]] = newfes[j]
            end
        end
    end


    if !has_regressors
        return FactorResult(esample, augmentdf, rss, iterations, converged)
    else
        return InteractiveFixedEffectModel(fs.b, matrix_vcov, vcov, esample, augmentdf, 
            coef_names, yname, f, nobs, dof_residual, r2, r2_a, r2_within, 
            rss, sum(iterations), all(converged))
    end
end