# Object constructed by the user
type PanelFactorModel
    id::Symbol
    time::Symbol
    rank::Int64
end

# object returned by fitting variable
type PanelFactorResult 
    id::PooledDataVector
    time::PooledDataVector
    loadings::Matrix{Float64}  # N x d
    factors::Matrix{Float64} # T x d
    iterations::Vector{Int64}
    iteration_converged::Vector{Bool}
    x_converged::Vector{Bool}
    f_converged::Vector{Bool}
    gr_converged::Vector{Bool}
end

# Object returned when fitting model
type PanelFactorModelResult 
    id::PooledDataVector
    time::PooledDataVector
    coef::Vector{Float64} 
    loadings::Matrix{Float64}  # N x d
    factors::Matrix{Float64} # T x d
    iterations::Int64
    converged::Bool
end

