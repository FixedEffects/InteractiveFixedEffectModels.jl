##############################################################################
##
## Object constructed by the user
##
##############################################################################

# Object constructed by the user
struct InteractiveFixedEffectFormula
    id::Union{Symbol, Expr}
    time::Union{Symbol, Expr}
    rank::Int64
end
function InteractiveFixedEffectFormula(arg)
    arg.head == :tuple && length(arg.args) == 2 || throw("@ife does not have a correct syntax")
    arg1 = arg.args[1]
    arg2 = arg.args[2]
    arg1.head == :call && arg1.args[1] == :+ && length(arg1.args) == 3 || throw("@ife does not have a correct syntax")
    InteractiveFixedEffectFormula(arg1.args[2], arg1.args[3], arg2)
end


abstract type AbstractFactorModel{T} end
abstract type AbstractFactorSolution{T} end
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

rank(::FactorModel{Rank}) where {Rank} = Rank

struct FactorSolution{Rank, Tid, Ttime} <: AbstractFactorSolution{Rank}
    idpool::Tid
    timepool::Ttime
end

function FactorSolution(idpool::Tid, timepool::Ttime) where {Tid, Ttime}
    r = size(idpool, 2)
    @assert r == size(timepool, 2)
    FactorSolution{r, Tid, Ttime}(idpool, timepool)
end

function view(f::AbstractFactorSolution, I::Union{AbstractArray,Colon,Int64}...)
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


function DataFrame(fp::AbstractFactorModel, fs::AbstractFactorSolution, esample::AbstractVector{Bool})
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

rank(::InteractiveFixedEffectsModel{Rank}) where {Rank} = Rank

function convert(::Type{FactorModel}, f::InteractiveFixedEffectsModel{Rank, W, Rid, Rtime}) where {Rank, W, Rid, Rtime}
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
convert(::Type{FactorSolution}, f::InteractiveFixedEffectsSolution) = FactorSolution(f.idpool, f.timepool)

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

    esample::BitVector      # Is the row of the original dataframe part of the estimation sample?
    augmentdf::DataFrame

    coefnames::Vector       # Name of coefficients
    yname::Symbol           # Name of dependent variable
    formula::FormulaTerm        # Original formula 

    nobs::Int64             # Number of observations
    dof_residual::Int64      # degree of freedoms

    r2::Float64             # R squared
    adjr2::Float64           # R squared adjusted
    r2_within::Float64      # R within

    rss::Float64
    iterations::Int         # Number of iterations        
    converged::Bool         # Has the demeaning algorithm converged?

end
coef(x::InteractiveFixedEffectModel) = x.coef
coefnames(x::InteractiveFixedEffectModel) = x.coefnames
vcov(x::InteractiveFixedEffectModel) = x.vcov
nobs(x::InteractiveFixedEffectModel) = x.nobs
dof_residual(x::InteractiveFixedEffectModel) = x.dof_residual
r2(x::InteractiveFixedEffectModel) = x.r2
adjr2(x::InteractiveFixedEffectModel) = x.adjr2
islinear(x::InteractiveFixedEffectModel) = true
rss(x::InteractiveFixedEffectModel) = x.rss
predict(::InteractiveFixedEffectModel, ::AbstractDataFrame) = error("predict is not defined for linear factor models. Use the option save = true")
residuals(::InteractiveFixedEffectModel, ::AbstractDataFrame) = error("residuals is not defined for linear factor models. Use the option save = true")
function confint(x::InteractiveFixedEffectModel)
    scale = quantile(TDist(dof_residual(x)), 1 - (1-0.95)/2)
    se = stderror(x)
    hcat(x.coef -  scale * se, x.coef + scale * se)
end

title(::InteractiveFixedEffectModel) = "Linear Factor Model"
top(x::InteractiveFixedEffectModel) = [
            "Number of obs" sprint(show, nobs(x); context=:compact => true);
            "Degree of freedom" sprint(show, nobs(x) - dof_residual(x); context=:compact => true);
            "R2"  @sprintf("%.3f", x.r2);
            "R2 within"  @sprintf("%.3f", x.r2_within);
            "Iterations" sprint(show, x.iterations; context=:compact => true);
            "Converged" sprint(show, x.converged; context=:compact => true)
            ]

function Base.show(io::IO, x::InteractiveFixedEffectModel)
    show(io, coeftable(x))
end
function coeftable(x::InteractiveFixedEffectModel)
    ctitle = title(x)
    ctop = top(x)
    cc = coef(x)
    se = stderror(x)
    coefnms = coefnames(x)
    conf_int = confint(x)
    # put (intercept) last
    if !isempty(coefnms) && ((coefnms[1] == Symbol("(Intercept)")) || (coefnms[1] == "(Intercept)"))
        newindex = vcat(2:length(cc), 1)
        cc = cc[newindex]
        se = se[newindex]
        conf_int = conf_int[newindex, :]
        coefnms = coefnms[newindex]
    end
    tt = cc ./ se
    FixedEffectModels.CoefTable2(
        hcat(cc, se, tt, ccdf.(Ref(FDist(1, dof_residual(x))), abs2.(tt)), conf_int[:, 1:2]),
        ["Estimate","Std.Error","t value", "Pr(>|t|)", "Lower 95%", "Upper 95%" ],
        ["$(coefnms[i])" for i = 1:length(cc)], 4, ctitle, ctop)
end
