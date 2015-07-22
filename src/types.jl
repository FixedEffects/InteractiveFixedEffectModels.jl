# Object constructed by the user
type PanelFactorModel
    id::Symbol
    time::Symbol
    rank::Int64
end

# object returned by fitting variable
abstract AbstractPanelFactorResult

type PanelFactorResult{Tid, Ttime} <: AbstractPanelFactorResult
    id::PooledDataVector{Tid}
    time::PooledDataVector{Ttime}
    loadings::Matrix{Float64}  # N x d
    factors::Matrix{Float64} # T x d
    iterations::Vector{Int64}
    iteration_converged::Vector{Bool}
    x_converged::Vector{Bool}
    f_converged::Vector{Bool}
    gr_converged::Vector{Bool}
end


# Object returned when fitting model
type PanelFactorModelResult{Tid, Ttime} <: AbstractPanelFactorResult
    coef::Vector{Float64}
    id::PooledDataVector{Tid}
    time::PooledDataVector{Ttime}
    loadings::Matrix{Float64}  # N x d
    factors::Matrix{Float64} # T x d
    iterations::Int64
    iteration_converged::Bool
    x_converged::Bool
    f_converged::Bool
    gr_converged::Bool
end


function normalize!(x::AbstractPanelFactorResult)
    res_matrix = A_mul_Bt(x.loadings, x.factors)
    variance = At_mul_B(res_matrix, res_matrix)
    F = eigfact!(variance)
    factors = sub(F[:vectors], :, (size(x.factors, 1) - size(x.factors, 2) + 1):size(x.factors, 1))
    newfactors = Array(Float64, (size(x.factors, 1), size(x.factors, 2)))
    for j in 1:size(x.factors, 2)
        x.factors[:, j] = factors[:, size(x.factors, 2) + 1 - j]
    end
    scale!(x.factors, sqrt(size(x.factors, 1)))
    A_mul_B!(x.loadings, res_matrix, x.factors)
    x.loadings = scale!(x.loadings, 1/size(x.factors, 1))
end
