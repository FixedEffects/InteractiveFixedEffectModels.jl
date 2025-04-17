##############################################################################
##
## Iterate on terms
##
##############################################################################

eachterm(@nospecialize(x::AbstractTerm)) = (x,)
eachterm(@nospecialize(x::NTuple{N, AbstractTerm})) where {N} = x

##############################################################################
##
## Parse FixedEffect
##
##############################################################################
fesymbol(t::FixedEffectModels.FixedEffectTerm) = t.x
fesymbol(t::FunctionTerm{typeof(fe)}) = Symbol(t.args[1])

function parse_fixedeffect(df, formula)
    if isdefined(FixedEffectModels, :parse_fe)
        formula, formula_fes = FixedEffectModels.parse_fe(formula)
        fes, ids, fekeys = FixedEffectModels.parse_fixedeffect(df, formula_fes)
    else
        fes, ids, fekeys, formula = FixedEffectModels.parse_fixedeffect(df, formula)
    end
    return fes, ids, fekeys, formula
end