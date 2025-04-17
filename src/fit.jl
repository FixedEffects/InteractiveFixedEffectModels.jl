

function regife(
    @nospecialize(df), 
    @nospecialize(formula::FormulaTerm),
    @nospecialize(vcov::CovarianceEstimator = Vcov.simple());
    @nospecialize(weights::Union{Symbol, Nothing} = nothing), 
    @nospecialize(subset::Union{AbstractVector, Nothing} = nothing),
    @nospecialize(method::Symbol = :dogleg), 
    @nospecialize(lambda::Number = 0.0), 
    @nospecialize(maxiter::Integer = 10_000), 
    @nospecialize(tol::Real = 1e-9), 
    @nospecialize(save::Union{Bool, Nothing} = nothing),
    @nospecialize(contrasts::Dict = Dict{Symbol, Any}()))

    ##############################################################################
    ##
    ## Transform DataFrame -> Matrix
    ##
    ##############################################################################
    formula_origin = formula

    df = DataFrame(df; copycols = false)
    if  (ConstantTerm(0) ∉ eachterm(formula.rhs)) & (ConstantTerm(1) ∉ eachterm(formula.rhs))
        formula = FormulaTerm(formula.lhs, tuple(ConstantTerm(1), eachterm(formula.rhs)...))
    end


    m, formula = parse_interactivefixedeffect(df, formula)
    has_weights = (weights != nothing)


    ## create a dataframe without missing values & negative weightss
    vars = StatsModels.termvars(formula)
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
     # Compute weights
     if has_weights
         weights = Weights(convert(Vector{Float64}, view(df, esample, weights)))
         sqrtw = sqrt.(weights)
     else
         weights = uweights(sum(esample))
         sqrtw = ones(length(weights))
     end
    for a in eachterm(formula.rhs)
       if has_fe(a)
           isa(a, InteractionTerm) && error("Fixed effects cannot be interacted")
           Symbol(fesymbol(a)) ∉ factor_vars && error("FixedEffect should correspond to id or time dimension of the factor model")
       end
    end
    fes, ids, fekeys, formula = parse_fixedeffect(df, formula)
    has_fes = !isempty(fes)
    has_fes_intercept = false
    ## Compute factors, an array of AbtractFixedEffects
    if has_fes
        if any([isa(fe.interaction, UnitWeights) for fe in fes])
                formula = FormulaTerm(formula.lhs, tuple(ConstantTerm(0), (t for t in eachterm(formula.rhs) if t!= ConstantTerm(1))...))
                has_fes_intercept = true
        end
        fes = FixedEffect[fe[esample] for fe in fes]
        feM = AbstractFixedEffectSolver{Float64}(fes, weights, Val{:cpu})
    end


    has_intercept = ConstantTerm(1) ∈ eachterm(formula.rhs)


    iterations = 0
    converged = false
    # get two dimensions

    id = GroupedArray(df[esample, m.id])
    time = GroupedArray(df[esample, m.time])

    ##############################################################################
    ##
    ## Construct vector y and matrix X
    ##
    ##############################################################################
    subdf = Tables.columntable((; (x => disallowmissing(view(df[!, x], esample)) for x in unique(vcat(vars)))...))
                
    formula_schema = apply_schema(formula, schema(formula, subdf, contrasts), StatisticalModel)

    y = convert(Vector{Float64}, response(formula_schema, subdf))
    tss_total = tss(y, has_intercept || has_fes_intercept, weights)

    X = convert(Matrix{Float64}, modelmatrix(formula_schema, subdf))

    # change default if has_regressors
    has_regressors = size(X, 2) > 0
    if save == nothing 
        save = !has_regressors
    end



    # Compute demeaned X
    responsename, coef_names = coefnames(formula_schema)
    if !isa(coef_names, Vector)
        coef_names = [coef_names]
    end
    responsename = Symbol(responsename)
    coef_names = Symbol.(coef_names)


    # demean variables
    if has_fes
        solve_residuals!(y, feM)
        solve_residuals!(X, feM, progress_bar = false)
    end
    
 

    ##############################################################################
    ##
    ## Estimate Model on Matrix
    ##
    ##############################################################################

    # initialize factor models at 0.1
    idpool = fill(0.1, id.ngroups, m.rank)
    timepool = fill(0.1, time.ngroups, m.rank)
  
    y .= y .* sqrtw 
    X .= X .* sqrtw
    if !has_regressors
        # factor model 
        fp = FactorModel(y, sqrtw, id.groups, time.groups, m.rank)
        fs = FactorSolution(idpool, timepool)
        (fs, iterations, converged) = 
            fit!(Val{method}, fp, fs; maxiter = maxiter, tol = tol, lambda = lambda)
    else 
        # interactive fixed effect
        coef = X \ y
        fp = FactorModel(y - X * coef, sqrtw, id.groups, time.groups, m.rank)
        fs = FactorSolution(idpool, timepool)
        fit!(Val{:levenberg_marquardt}, fp, fs; maxiter = 100, tol = 1e-3, lambda = lambda)

        fs = InteractiveFixedEffectsSolution(coef, fs.idpool, fs.timepool)
        fp = InteractiveFixedEffectsModel(y, sqrtw, X, id.groups, time.groups, m.rank)


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
            newfeM = AbstractFixedEffectSolver{Float64}(getfactors(fp, fs), weights, Val{:cpu})
            ym .= ym ./ sqrtw
            solve_residuals!(ym, newfeM, tol = tol, maxiter = maxiter)
            ym .= ym .* sqrtw
            Xm .= Xm ./ sqrtw
            solve_residuals!(Xm, newfeM, tol = tol, maxiter = maxiter)
            Xm .= Xm .* sqrtw
            ydiff = Xm  * (fs.b - Xm \ ym)
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
    fp = FactorModel(copy(y), sqrtw, id.groups, time.groups, m.rank)
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
        vcov_data = Vcov.VcovData(Xm, crossxm, inv(crossxm), residualsm, dof_residual)
        matrix_vcov = StatsBase.vcov(vcov_data, vcov_method_data)
        # compute various r2
        nobs = sum(esample)
        rss = sum(abs2, residualsm)
        _tss = tss(ym ./ sqrtw, has_intercept || has_fes_intercept, weights)
        r2_within = 1 - rss / _tss 

        rss = sum(abs2, residuals)
        r2 = 1 - rss / tss_total 
        r2_a = 1 - rss / tss_total * (nobs - has_intercept) / dof_residual 
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
            fp = FactorModel(oldresiduals, sqrtw, id.groups, time.groups, m.rank)
            subtract_factor!(fp, fs)
            axpy!(-1.0, residuals, oldresiduals)
            # get fixed effect
            newfes, b, c = solve_coefficients!(oldresiduals, feM; tol = tol, maxiter = maxiter)
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
            coef_names, responsename, formula_origin, formula, nobs, dof_residual, rss, tss_total, r2, r2_a, r2_within, sum(iterations), all(converged))
    end
end
