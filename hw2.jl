using JuliaFormatter
using LinearAlgebra
using Printf, Lazy
include("./NewtonsMethod.jl")
using .NewtonsMethod

format_file("hw2.jl")
format_file("NewtonsMethod.jl")
const p¹¹(a¹) = a¹ / 2
const p¹²(a¹) = 1 - a¹ / 2
const p²¹(a²) = 0
const p²²(a²) = 1
const p(a¹, a²) = [p¹¹(a¹) p¹²(a¹); p²¹(a²) p²²(a²)]
const r¹(a) = -a^2
const γ = 0.9
const ε = 10^-16


function T¹(a¹, U¹, U²)
    #@assert 0 <= a¹ <= 2
    -a¹^2 + γ * p¹¹(a¹) * U¹ + γ * p¹²(a¹) * U²
end

function dT¹da¹(a¹, U¹, U²)
    -2 * a¹ + γ * U¹ / 2 - γ * U² / 2
end



function T²(a², U¹, U²)
    -0.5 + γ * p²¹(a²) * U¹ + γ * p²²(a²) * U²
end

const J(a¹, a²) = (I - γ * p(a¹, a²)) \ [r¹(a¹); -0.5]
#const J(a¹,a²) = inv((I - γ * p(a¹,a²))) * [r¹(a¹);-0.5]
const J²(a¹, a²) = J(a¹, a²)[2]
const J¹(a¹, a²) = J(a¹, a²)[1]
#const J¹(a¹,a²) = (1 - γ * p¹¹(a¹))^(-1) * (r¹(a¹) + γ * p¹²(a¹) * J²(a¹,a²))
#const argmaxT¹(U¹, U²) = @> (γ * U¹ / 4 - γ * U² / 4) min(2) max(0)
function argmaxT¹(U¹, U²)
    f = a¹ -> dT¹da¹(a¹, U¹, U²)
    dfdx = a¹ -> -2
    x = NewtonsMethod.newtons_method(1, f, dfdx)
    return x
end

const argmaxT²(U¹, U²) = 0

# Value iteration 

function vi_opt_eval(Vₓ¹, Vₓ²)
    aₓ₊₁¹ = argmaxT¹(Vₓ¹, Vₓ²)
    aₓ₊₁² = argmaxT²(Vₓ¹, Vₓ²)
    Vₓ₊₁¹ = T¹(aₓ₊₁¹, Vₓ¹, Vₓ²)
    Vₓ₊₁² = T²(aₓ₊₁², Vₓ¹, Vₓ²)
    error = max(abs(Vₓ₊₁¹ - Vₓ¹), abs(Vₓ₊₁² - Vₓ²))
    return Vₓ₊₁¹, Vₓ₊₁², aₓ₊₁¹, aₓ₊₁², error
end

print_vi_res((x, Vₓ¹, Vₓ², aₓ¹, aₓ², error)) = @printf(
    "x = %-6i   Vₓ¹ = %-22.16f  Vₓ² = %-22.16f aₓ¹ = %-20.16f  aₓ² = %-20.16f error = %-20.16f\n",
    x,
    Vₓ¹,
    Vₓ²,
    aₓ¹,
    aₓ²,
    error
)

function value_iteration(V₀¹, V₀², maxiterations = 50)
    Vₓ¹ = V₀¹
    Vₓ² = V₀²
    print_vi_res((0, Vₓ¹, Vₓ², NaN, NaN, NaN))
    for m = 0:(maxiterations-1)
        Vₓ₊₁¹, Vₓ₊₁², aₓ₊₁¹, aₓ₊₁², error = vi_opt_eval(Vₓ¹, Vₓ²)
        print_vi_res((m + 1, Vₓ₊₁¹, Vₓ₊₁², aₓ₊₁¹, aₓ₊₁², error))
        if error < ε * (1 - γ) / (2 * γ)
            break
        else
            Vₓ¹ = Vₓ₊₁¹
            Vₓ² = Vₓ₊₁²
        end
    end
end

# The following will iteratively map (m, Vₘ¹, Vₘ², aₘ¹, error) to (m+1, Vₘ₊₁¹, Vₘ₊₁², aₘ₊₁¹, error)
vi_step((m, Vₓ¹, Vₓ², _, _, _)) = m + 1, vi_opt_eval(Vₓ¹, Vₓ²)...
vi_stopping_cond((_, _, _, _, _, error)) = error < ε * (1 - γ) / (2 * γ)
value_iteration_lazy(V₀¹, V₀²) =
    @>> (0, V₀¹, V₀², NaN, NaN, NaN) iterated(vi_step) takeuntil(vi_stopping_cond)

# These accomplish the same thing
println("Value iteration:")
value_iteration(J¹(0, 0), J²(0, 0))
println("Value iteration:")
@>> value_iteration_lazy(J¹(0, 0), J²(0, 0)) foreach(print_vi_res)


#Policy iteration 

function pi_opt_eval(aₓ¹, aₓ²)
    Vₓ¹ = J¹(aₓ¹, aₓ²)
    Vₓ² = J²(aₓ¹, aₓ²)
    aₓ₊₁¹ = argmaxT¹(Vₓ¹, Vₓ²)
    aₓ₊₁² = aₓ²
    error = max(abs(aₓ₊₁¹ - aₓ¹), abs(aₓ₊₁² - aₓ²))
    return Vₓ¹, Vₓ², aₓ₊₁¹, aₓ₊₁², error
end

print_pi_res((x, Vₓ₋₁¹, Vₓ₋₁², aₓ¹, aₓ², error)) = @printf(
    "x = %-6i   aₓ¹ = %-20.16f  aₓ² = %-20.16f Vₓ₋₁¹ = %-20.16f  Vₓ₋₁² = = %-20.16f error = %-20.16f\n",
    x,
    aₓ¹,
    aₓ²,
    Vₓ₋₁¹,
    Vₓ₋₁²,
    error
)

function policy_iteration(a₀¹, a₀², maxiterations = 50)
    Vₓ₋₁² = NaN
    Vₓ₋₁¹ = NaN
    aₓ¹ = a₀¹
    aₓ² = a₀²
    print_pi_res((0, NaN, NaN, a₀¹, a₀², NaN))
    for x = 0:(maxiterations-1)
        Vₓ¹, Vₓ², aₓ₊₁¹, aₓ₊₁², error = pi_opt_eval(aₓ¹, aₓ²)
        print_pi_res((x + 1, Vₓ¹, Vₓ², aₓ₊₁¹, aₓ₊₁², error))
        if error == 0
            break
        else
            aₓ¹ = aₓ₊₁¹
            aₓ² = aₓ₊₁²
            Vₓ₋₁¹ = Vₓ¹
            Vₓ₋₁² = Vₓ²
        end
    end
end

# The following will iteratively map (m, Vₘ₋₁¹, Vₘ₋₁², aₘ¹, error) to (m+1, Vₘ¹, Vₘ², aₘ₊₁¹, error)
pi_step((x, _, _, aₓ¹, aₓ², _)) = x + 1, pi_opt_eval(aₓ¹, aₓ²)...
pi_stopping_cond((_, _, _, _, _, error)) = error == 0.0
policy_iteration_lazy(a₀¹, a₀²) =
    @>> (0, NaN, NaN, a₀¹, a₀², NaN) iterated(pi_step) takeuntil(pi_stopping_cond)



# And so do these
println("Policy iteration:")
policy_iteration(0.0, 0.0)
println("Policy iteration:")
@>> policy_iteration_lazy(0.0, 0.0) foreach(print_pi_res)



# modified iteration
function vk_opt_eval(Vₓ¹, Vₓ², k)
    aₓ₊₁¹ = argmaxT¹(Vₓ¹, Vₓ²)
    aₓ₊₁² = argmaxT²(Vₓ¹, Vₓ²)
    Vₓ₊₁¹ = T¹(aₓ₊₁¹, Vₓ¹, Vₓ²)
    Vₓ₊₁² = T²(aₓ₊₁², Vₓ¹, Vₓ²)
    for i = 2:k
        Vₓ₊₁¹ = T¹(aₓ₊₁¹, Vₓ₊₁¹, Vₓ₊₁²)
        Vₓ₊₁² = T²(aₓ₊₁², Vₓ₊₁¹, Vₓ₊₁²)
    end
    error = max(abs(Vₓ₊₁¹ - Vₓ¹), abs(Vₓ₊₁² - Vₓ²))
    return Vₓ₊₁¹, Vₓ₊₁², aₓ₊₁¹, aₓ₊₁², error
end

print_vk_res((x, Vₓ¹, Vₓ², aₓ¹, aₓ², error)) = @printf(
    "x = %-6i   Vₓ¹ = %-22.16f  Vₓ² = %-22.16f aₓ¹ = %-20.16f  aₓ² = %-20.16f  error = %-20.16f\n",
    x,
    Vₓ¹,
    Vₓ²,
    aₓ¹,
    aₓ²,
    error
)

function modified_iteration(V₀¹, V₀², k, maxiterations = 50)
    Vₓ¹ = V₀¹
    Vₓ² = V₀²
    print_vk_res((0, Vₓ¹, Vₓ², NaN, NaN, NaN))
    for m = 0:(maxiterations-1)
        Vₓ₊₁¹, Vₓ₊₁², aₓ₊₁¹, aₓ₊₁², error = vk_opt_eval(Vₓ¹, Vₓ², k)
        print_vk_res((m + 1, Vₓ₊₁¹, Vₓ₊₁², aₓ₊₁¹, aₓ₊₁², error))
        if error < ε * (1 - γ) / (2 * γ)
            return m + 1
            break
        else
            Vₓ¹ = Vₓ₊₁¹
            Vₓ² = Vₓ₊₁²
        end
    end
end

for i = 2:100
    m = modified_iteration(J¹(0, 0), J²(0, 0), i)
    if m <= 6
        println(i)
        break
    end
end



#  NewtonsMethod
g(x) = x^2 - 2 * x - 3
dgdx(x) = 2 * x - 2
NewtonsMethod.newtons_method(2, g, dgdx)
NewtonsMethod.newtons_method(0, g, dgdx)
