using JuliaFormatter
using LinearAlgebra
using Printf, Lazy
using Plots
using SparseArrays

format_file("hw3.jl")
include("./explicit.jl")
using .ExplicitSolution

const x_min = 0
const N = 2000 #1000
const α₀ = zeros(N + 1)

function x⁺(t)
    max(0, t)
end
function x⁻(t)
    min(0, t)
end


function Aᵅ(μ, b, α, x_max)
    h = x_max / N
    a = spzeros(N + 1, N + 1)
    a[1, 1] = b^2 / (2 * h^2)
    #a[1, 1] = -abs(μ - α[1]) / h - (b / h)^2
    #a[1, 2] = x⁺(μ - α[1]) / h + b^2 / (2 * h^2)
    a[N+1, N+1] = -b^2 / (2 * h^2) + x⁻(μ - α[N]) / h
    a[N+1, N] = b^2 / (2 * h^2) - x⁻(μ - α[N]) / h
    for i = 2:N
        a[i, i] = -abs(μ - α[i]) / h - (b / h)^2
        a[i, i-1] = -x⁻(μ - α[i]) / h + b^2 / (2 * h^2)
        a[i, i+1] = x⁺(μ - α[i]) / h + b^2 / (2 * h^2)
    end
    return a
end

function B(μ, b, α, x_max)
    h = x_max / N
    C = zeros(N + 1)
    C[N+1] = b^2 / (2 * h) + x⁺(μ - α[N])

    return C
end

# Policy Iteration for V_K

function argmaxT(U, K, h)
    α_new = zeros(N + 1)
    #α_new[1] = 0
    ∂U = zeros(N + 1)
    #∂U[1] = U[1] / h
    for i = 2:N+1
        ∂U[i] = (U[i] - U[i-1]) / h
        if ∂U[i] < 1
            α_new[i] = K
        end
    end
    #=for i = 2:N+1
        if ∂U[i] < 1
            α_new[i] = K
        end
    end=#
    return α_new
end

function J(μ, r, b, α, x_max)
    A = Aᵅ(μ, b, α, x_max)
    h = x_max / N
    B₁ = B(μ, b, α, x_max)

    return (r * sparse(I, N + 1, N + 1) - A) \ (B₁ + α)
end

function pi_opt_eval(μ, r, b, α, x_max, K)
    V₁ = J(μ, r, b, α, x_max)
    h = x_max / N
    α₁ = argmaxT(V₁, K, h)
    error = maximum(abs.(α - α₁))
    #V₁, α₁, 
    return V₁, α₁, error
end

function policy_iteration(μ, r, b, α₀, x_max, K, maxiterations = 50)
    Vₒ = NaN
    αₒ = α₀
    for o = 0:(maxiterations-1)
        Vₒ₊₁, αₒ₊₁, error = pi_opt_eval(μ, r, b, αₒ, x_max, K)
        #@printf("error=%-20.16f\n", error)
        if error == 0
            break
        else
            αₒ = αₒ₊₁
            Vₒ = Vₒ₊₁
        end
    end
    return Vₒ
end
println("Policy iteration:")

#policy_iteration(0.1, 0.03, 0.1, α₀, 3, 100000)
f = ExplicitSolution.V(0.03, 0.1, 0.1)
Kmax = 10000
Kmin = 50
err = zeros(200)
err_1 = zeros(200)
err_2 = zeros(200)

V_inf = f.([3 * (i - 1) / N for i = 1:N+1])
V_inf_1 = f.([4 * (i - 1) / N for i = 1:N+1])
V_inf_2 = f.([5 * (i - 1) / N for i = 1:N+1])
for K = Kmin:50:Kmax
    u_k = policy_iteration(0.1, 0.03, 0.1, α₀, 3, K)
    u_k1 = policy_iteration(0.1, 0.03, 0.1, α₀, 4, K)
    u_k2 = policy_iteration(0.1, 0.03, 0.1, α₀, 5, K)
    index = Int((K - Kmin) / 50 + 1)
    err[index] = sum((u_k - V_inf) .^ 2) / N
    err_1[index] = sum((u_k1 - V_inf_1) .^ 2) / N
    err_2[index] = sum((u_k2 - V_inf_2) .^ 2) / N
end

plot(
    log.(Kmin:50:Kmax),
    [log.(err), log.(err_1), log.(err_2)],
    label = ["xmax=3" "xmax=4" "xmax=5"],
)

kmax = 50
kmin = 5
err_3 = zeros(46)
err_4 = zeros(46)
err_5 = zeros(46)
for K = kmin:kmax
    u_k = policy_iteration(0.1, 0.03, 0.1, α₀, 3, K)
    u_k1 = policy_iteration(0.1, 0.03, 0.1, α₀, 4, K)
    u_k2 = policy_iteration(0.1, 0.03, 0.1, α₀, 5, K)
    index = K - kmin + 1
    err_3[index] = sum((u_k - V_inf) .^ 2) / N
    err_4[index] = sum((u_k1 - V_inf_1) .^ 2) / N
    err_5[index] = sum((u_k2 - V_inf_2) .^ 2) / N
end

plot(
    log.(kmin:kmax),
    [log.(err_3), log.(err_4), log.(err_5)],
    label = ["xmax=3" "xmax=4" "xmax=5"],
)


k1 = 1
k2 = 3
Vk1 = policy_iteration(0.1, 0.03, 0.1, α₀, 4, k1)
Vk2 = policy_iteration(0.1, 0.03, 0.1, α₀, 4, k2)
plot(1:N+1, [Vk1, Vk2, V_inf_1], label = ["V1" "V3" "Vinf"])
