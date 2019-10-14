


function evaluate_subset(df, ex::Expr)
    if ex.head == :call
        return Expr(ex.head, ex.args[1], (evaluate_subset(df, ex.args[i]) for i in 2:length(ex.args))...)
    else
        return Expr(ex.head, (evaluate_subset(df, ex.args[i]) for i in 1:length(ex.args))...)
    end
end
evaluate_subset(df, ex::Symbol) = df[!, ex]
evaluate_subset(df, ex)  = ex