##############################################################################
##
## Iterate on terms
##
##############################################################################

eachterm(@nospecialize(x::AbstractTerm)) = (x,)
eachterm(@nospecialize(x::NTuple{N, AbstractTerm})) where {N} = x
TermOrTerms = Union{AbstractTerm, NTuple{N, AbstractTerm} where N}

##############################################################################
##
## Parse FixedEffect
##
##############################################################################
fesymbol(t::FixedEffectModels.FixedEffectTerm) = t.x
fesymbol(t::FunctionTerm{typeof(fe)}) = Symbol(t.args_parsed[1])
