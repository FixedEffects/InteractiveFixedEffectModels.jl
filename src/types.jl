##############################################################################
##
## Object constructed by the user
##
##############################################################################
function ife(id::Term, time::Term, rank::Int) 
    InteractiveFixedEffectTerm(Symbol(id), Symbol(time), rank)
end


# Object constructed by the user
struct InteractiveFixedEffectTerm <: AbstractTerm
    id::Symbol
    time::Symbol
    rank::Int
end

has_ife(x::InteractiveFixedEffectTerm) = true
has_ife(x::FunctionTerm{typeof(ife)}) = true
has_ife(x::AbstractTerm) = false
function parse_interactivefixedeffect(df::AbstractDataFrame, formula::FormulaTerm)
    m = nothing
    for term in FixedEffectModels.eachterm(formula.rhs)
        if term isa FunctionTerm{typeof(ife)}
            m = InteractiveFixedEffectTerm(term.args[1].sym, term.args[2].sym, term.args[3].n)
        elseif term isa InteractiveFixedEffectTerm
            m = term
        end
    end
    return m, FormulaTerm(formula.lhs, tuple((term for term in FixedEffectModels.eachterm(formula.rhs) if !has_ife(term))...))
end

abstract type AbstractFactorModel{T} end
abstract type AbstractFactorSolution{T} end

# to deprecate
function OldInteractiveFixedEffectFormula(arg)
    arg.head == :tuple && length(arg.args) == 2 || throw("ife does not have a correct syntax")
    arg1 = arg.args[1]
    arg2 = arg.args[2]
    arg1.head == :call && arg1.args[1] == :+ && length(arg1.args) == 3 || throw("ife does not have a correct syntax")
    InteractiveFixedEffectTerm(arg1.args[2], arg1.args[3], arg2)
end


##############################################################################
##
## Factor Model
##
##############################################################################

struct FactorModel{Rank, W, Rid, Rtime} <: AbstractFactorModel{Rank}
    y::Vector{Float64}
    sqrtw::W
    idrefs::Vector{Rid}
    timerefs::Vector{Rtime}
end

function FactorModel(y::Vector{Float64}, sqrtw::W, idrefs::Vector{Rid}, timerefs::Vector{Rtime}, rank::Int) where {W, Rid, Rtime}
    FactorModel{rank, W, Rid, Rtime}(y, sqrtw, idrefs, timerefs)
end

LinearAlgebra.rank(::FactorModel{Rank}) where {Rank} = Rank

struct FactorSolution{Rank, Tid, Ttime} <: AbstractFactorSolution{Rank}
    idpool::Tid
    timepool::Ttime
end

function FactorSolution(idpool::Tid, timepool::Ttime) where {Tid, Ttime}
    r = size(idpool, 2)
    @assert r == size(timepool, 2)
    FactorSolution{r, Tid, Ttime}(idpool, timepool)
end

function Base.view(f::AbstractFactorSolution, I::Union{AbstractArray,Colon,Int64}...)
    FactorSolution(view(f.idpool, I...), view(f.timepool, I...))
end


## subtract_factor! and subtract_b!
function subtract_factor!(fm::AbstractFactorModel, fs::AbstractFactorSolution)
    for r in 1:rank(fm)
        subtract_factor!(fm, view(fs, :, r))
    end
end

function subtract_factor!(fm::AbstractFactorModel, fs::FactorSolution{1})
    @inbounds @simd for i in 1:length(fm.y)
        fm.y[i] -= fm.sqrtw[i] * fs.idpool[fm.idrefs[i]] * fs.timepool[fm.timerefs[i]]
    end
end


## rescale a factor model
function reverse(m::Matrix{R}) where {R}
    out = similar(m)
    for j in 1:size(m, 2)
        invj = size(m, 2) + 1 - j 
        @inbounds @simd for i in 1:size(m, 1)
            out[i, j] = m[i, invj]
        end
    end
    return out
end
function rescale!(fs::FactorSolution{1})
    out = norm(fs.timepool)
    rmul!(fs.idpool, out)
    rmul!(fs.timepool, 1 / out)
end
# normalize factors and loadings so that F'F = Id, Lambda'Lambda diagonal
function rescale!(newfs::AbstractFactorSolution, fs::AbstractFactorSolution)
    U = eigen!(Symmetric(fs.timepool' * fs.timepool))
    sqrtDx = diagm(0 => sqrt.(abs.(U.values)))
    mul!(newfs.idpool,  fs.idpool,  U.vectors * sqrtDx)
    V = eigen!(Symmetric(newfs.idpool' * newfs.idpool))
    mul!(newfs.idpool, fs.idpool, reverse(U.vectors * sqrtDx * V.vectors))
    mul!(newfs.timepool, fs.timepool, reverse(U.vectors * (sqrtDx \ V.vectors)))
    return newfs
end

rescale(fs::FactorSolution) = rescale!(similar(fs), fs)


## Create dataframe from pooledfactors
function getfactors(fp::AbstractFactorModel, fs::AbstractFactorSolution)
    # partial out Y and X with respect to i.id x factors and i.time x loadings
    newfes = FixedEffect[]
    for r in 1:rank(fp)
        idfe = FixedEffect{typeof(fp.idrefs), Vector{Float64}}(fp.idrefs, fs.timepool[fp.timerefs, r], length(fs.idpool))
        push!(newfes, idfe)
        timefe = FixedEffect{typeof(fp.timerefs), Vector{Float64}}(fp.timerefs, fs.idpool[fp.idrefs, r], length(fs.timepool))
        push!(newfes, timefe)
    end
    # obtain the residuals and cross 
    return newfes
end


function DataFrames.DataFrame(fp::AbstractFactorModel, fs::AbstractFactorSolution, esample::AbstractVector{Bool})
    df = DataFrame()
    for r in 1:rank(fp)
        # loadings
        df[!, Symbol("loadings$r")] = Vector{Union{Float64, Missing}}(missing, length(esample))
        df[esample, Symbol("loadings$r")] = fs.idpool[:, r][fp.idrefs]

        df[!, Symbol("factors$r")] = Vector{Union{Float64, Missing}}(missing, length(esample))
        df[esample, Symbol("factors$r")] = fs.timepool[:, r][fp.timerefs]
    end
    return df
end



##############################################################################
##
## Interactive Fixed Effect Models
##
##############################################################################

struct InteractiveFixedEffectsModel{Rank, W, Rid, Rtime} <: AbstractFactorModel{Rank}
    y::Vector{Float64}
    sqrtw::W
    X::Matrix{Float64}
    idrefs::Vector{Rid}
    timerefs::Vector{Rtime}
end

function InteractiveFixedEffectsModel(y::Vector{Float64}, sqrtw::W, X::Matrix{Float64}, idrefs::Vector{Rid}, timerefs::Vector{Rtime}, rank::Int) where {W, Rid, Rtime}
    InteractiveFixedEffectsModel{rank, W, Rid, Rtime}(y, sqrtw, X, idrefs, timerefs)
end

LinearAlgebra.rank(::InteractiveFixedEffectsModel{Rank}) where {Rank} = Rank

function Base.convert(::Type{FactorModel}, f::InteractiveFixedEffectsModel{Rank, W, Rid, Rtime}) where {Rank, W, Rid, Rtime}
    FactorModel{Rank, W, Rid, Rtime}(f.y, f.sqrtw, f.idrefs, f.timerefs)
end


struct InteractiveFixedEffectsSolution{Rank, Tb, Tid, Ttime} <: AbstractFactorSolution{Rank}
    b::Tb
    idpool::Tid
    timepool::Ttime
end
function InteractiveFixedEffectsSolution(b::Tb, idpool::Tid, timepool::Ttime) where {Tb, Tid, Ttime}
    r = size(idpool, 2)
    r == size(timepool, 2) || throw("factors and loadings don't have same dimension")
    InteractiveFixedEffectsSolution{r, Tb, Tid, Ttime}(b, idpool, timepool)
end
Base.convert(::Type{FactorSolution}, f::InteractiveFixedEffectsSolution) = FactorSolution(f.idpool, f.timepool)

struct InteractiveFixedEffectsSolutionT{Rank, Tb, Tid, Ttime} <: AbstractFactorSolution{Rank}   
    b::Tb   
    idpool::Tid 
    timepool::Ttime 
end 
function InteractiveFixedEffectsSolutionT(b::Tb, idpool::Tid, timepool::Ttime) where {Tb, Tid, Ttime}   
    r = size(idpool, 1) 
    r == size(timepool, 1) || throw("factors and loadings don't have same dimension")   
    InteractiveFixedEffectsSolutionT{r, Tb, Tid, Ttime}(b, idpool, timepool)    
end

function rescale(fs::InteractiveFixedEffectsSolution)
    fss = FactorSolution(fs.idpool, fs.timepool)
    newfss = similar(fss)
    rescale!(newfss, fss)
    InteractiveFixedEffectsSolution(fs.b, newfss.idpool, newfss.timepool)
end




struct HalfInteractiveFixedEffectsModel{Rank, W, Rid, Rtime} <: AbstractFactorModel{Rank}
    y::Vector{Float64}
    sqrtw::W
    X::Matrix{Float64}
    idrefs::Vector{Rid}
    timerefs::Vector{Rtime}
    timepool::Matrix{Float64}
    size::Tuple{Int, Int}
end

function HalfInteractiveFixedEffectsModel(y::Vector{Float64}, sqrtw::W, X::Matrix{Float64}, idrefs::Vector{Rid}, timerefs::Vector{Rtime}, timepool::Matrix{Float64}, size, rank::Int) where {W, Rid, Rtime}
    HalfInteractiveFixedEffectsModel{rank, W, Rid, Rtime}(y, sqrtw, X, idrefs, timerefs, timepool, size)
end

struct HalfInteractiveFixedEffectsSolution{Rank, Tb, Tid} <: AbstractFactorSolution{Rank}
    b::Tb
    idpool::Tid
end

##############################################################################
##
## Results
##
##############################################################################'

struct FactorResult 
    esample::BitVector
    augmentdf::DataFrame

    rss::Float64
    iterations::Int64
    converged::Bool
end


# result
struct InteractiveFixedEffectModel <: RegressionModel
    coef::Vector{Float64}   # Vector of coefficients
    vcov::Matrix{Float64}   # Covariance matrix
    vcov_type::CovarianceEstimator 

    esample::BitVector      # Is the row of the original dataframe part of the estimation sample?
    augmentdf::DataFrame

    coefnames::Vector       # Name of coefficients
    responsename::Symbol           # Name of dependent variable
    formula::FormulaTerm        # Original formula
    formula_schema::FormulaTerm # Schema for predict

    nobs::Int64             # Number of observations
    dof_residual::Int64      # degree of freedoms

    rss::Float64
    tss::Float64
    r2::Float64             # R squared
    adjr2::Float64           # R squared adjusted
    r2_within::Float64      # R within

    iterations::Int         # Number of iterations        
    converged::Bool         # Has the demeaning algorithm converged?

end
StatsAPI.coef(x::InteractiveFixedEffectModel) = x.coef
StatsAPI.coefnames(x::InteractiveFixedEffectModel) = x.coefnames
StatsAPI.responsename(m::InteractiveFixedEffectModel) = m.responsename
StatsAPI.vcov(x::InteractiveFixedEffectModel) = x.vcov
StatsAPI.nobs(x::InteractiveFixedEffectModel) = x.nobs
StatsAPI.dof_residual(x::InteractiveFixedEffectModel) = x.dof_residual
StatsAPI.r2(x::InteractiveFixedEffectModel) = x.r2
StatsAPI.adjr2(x::InteractiveFixedEffectModel) = x.adjr2
StatsAPI.islinear(x::InteractiveFixedEffectModel) = false
StatsAPI.deviance(x::InteractiveFixedEffectModel) = x.tss
StatsAPI.rss(x::InteractiveFixedEffectModel) = x.rss
StatsAPI.mss(m::InteractiveFixedEffectModel) = deviance(m) - rss(m)
StatsModels.formula(m::InteractiveFixedEffectModel) = m.formula_schema


StatsBase.predict(::InteractiveFixedEffectModel, ::AbstractDataFrame) = error("predict is not defined for linear factor models. Use the option save = true")
StatsBase.residuals(::InteractiveFixedEffectModel, ::AbstractDataFrame) = error("residuals is not defined for linear factor models. Use the option save = true")
function StatsBase.confint(x::InteractiveFixedEffectModel; level::Real = 0.95)
    scale = tdistinvcdf(dof_residual(x), 1 - (1 - level) / 2)
    se = stderror(x)
    hcat(x.coef -  scale * se, x.coef + scale * se)
end
function StatsBase.coeftable(x::InteractiveFixedEffectModel; level = 0.95)
    cc = coef(x)
    se = stderror(x)
    coefnms = coefnames(x)
    conf_int = confint(x; level = level)
    # put (intercept) last
    if !isempty(coefnms) && ((coefnms[1] == Symbol("(Intercept)")) || (coefnms[1] == "(Intercept)"))
        newindex = vcat(2:length(cc), 1)
        cc = cc[newindex]
        se = se[newindex]
        conf_int = conf_int[newindex, :]
        coefnms = coefnms[newindex]
    end
    tt = cc ./ se
    FixedEffectModels.CoefTable(
        hcat(cc, se, tt, fdistccdf.(Ref(1), Ref(dof_residual(x)), abs2.(tt)), conf_int[:, 1:2]),
        ["Estimate","Std.Error","t value", "Pr(>|t|)", "Lower 95%", "Upper 95%" ],
        ["$(coefnms[i])" for i = 1:length(cc)], 4)
end

##############################################################################
##
## Display Result
##
##############################################################################

title(::InteractiveFixedEffectModel) = "Interactive Fixed Effect Model"
top(x::InteractiveFixedEffectModel) = [
            "Number of obs" sprint(show, nobs(x); context=:compact => true);
            "Degree of freedom" sprint(show, nobs(x) - dof_residual(x); context=:compact => true);
            "R²"  @sprintf("%.3f", x.r2);
            "R² within"  @sprintf("%.3f", x.r2_within);
            "Iterations" sprint(show, x.iterations; context=:compact => true);
            "Converged" sprint(show, x.converged; context=:compact => true)
            ]

import StatsBase: NoQuote, PValue
function Base.show(io::IO, m::InteractiveFixedEffectModel)
    ct = coeftable(m)
    #copied from show(iio,cf::Coeftable)
    cols = ct.cols; rownms = ct.rownms; colnms = ct.colnms;
    nc = length(cols)
    nr = length(cols[1])
    if length(rownms) == 0
        rownms = [lpad("[$i]",floor(Integer, log10(nr))+3) for i in 1:nr]
    end
    mat = [j == 1 ? NoQuote(rownms[i]) :
           j-1 == ct.pvalcol ? NoQuote(sprint(show, PValue(cols[j-1][i]))) :
           j-1 in ct.teststatcol ? TestStat(cols[j-1][i]) :
           cols[j-1][i] isa AbstractString ? NoQuote(cols[j-1][i]) : cols[j-1][i]
           for i in 1:nr, j in 1:nc+1]
    io = IOContext(io, :compact=>true, :limit=>false)
    A = Base.alignment(io, mat, 1:size(mat, 1), 1:size(mat, 2),
                       typemax(Int), typemax(Int), 3)
    nmswidths = pushfirst!(length.(colnms), 0)
    A = [nmswidths[i] > sum(A[i]) ? (A[i][1]+nmswidths[i]-sum(A[i]), A[i][2]) : A[i]
         for i in 1:length(A)]
    totwidth = sum(sum.(A)) + 2 * (length(A) - 1)


    #intert my stuff which requires totwidth
    ctitle = string(typeof(m))
    halfwidth = div(totwidth - length(ctitle), 2)
    print(io, " " ^ halfwidth * ctitle * " " ^ halfwidth)
    ctop = top(m)
    for i in 1:size(ctop, 1)
        ctop[i, 1] = ctop[i, 1] * ":"
    end
    println(io, '\n', repeat('=', totwidth))
    halfwidth = div(totwidth, 2) - 1
    interwidth = 2 +  mod(totwidth, 2)
    for i in 1:(div(size(ctop, 1) - 1, 2)+1)
        print(io, ctop[2*i-1, 1])
        print(io, lpad(ctop[2*i-1, 2], halfwidth - length(ctop[2*i-1, 1])))
        print(io, " " ^interwidth)
        if size(ctop, 1) >= 2*i
            print(io, ctop[2*i, 1])
            print(io, lpad(ctop[2*i, 2], halfwidth - length(ctop[2*i, 1])))
        end
        println(io)
    end
   
    # rest of coeftable code
    println(io, repeat('=', totwidth))
    print(io, repeat(' ', sum(A[1])))
    for j in 1:length(colnms)
        print(io, "  ", lpad(colnms[j], sum(A[j+1])))
    end
    println(io, '\n', repeat('─', totwidth))
    for i in 1:size(mat, 1)
        Base.print_matrix_row(io, mat, A, i, 1:size(mat, 2), "  ")
        i != size(mat, 1) && println(io)
    end
    println(io, '\n', repeat('=', totwidth))
    nothing
end
