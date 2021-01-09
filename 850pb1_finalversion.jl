using JuliaFormatter
using Distributions: cdf, Normal
using Plots
format_file("850pb1_finalversion.jl")
function eu_put(S, K, r, σ, t, N)
    Δt = t / N
    u = exp(σ * √Δt)
    d = 1 / u
    R = exp(r * Δt)
    p = (R - d) / (u - d)
    q = 1 - p
    tree = zeros((N + 1, N + 1))

    for k = N:-1:0
        for j = 0:k
            tree[j+1, k+1] = max(0, K - S * exp((2 * j - k) * σ * √Δt)) # binomial tree
        end
    end
    Z = tree[:, N+1]
    for n = N-1:-1:0
        for i = 0:n
            #x = K - S * exp((2 * i - n) * σ * √Δt) 
            y = (q * Z[i+1] + p * Z[i+2]) / R
            Z[i+1] = y
        end
    end

    return Z[1]
end

function am_put(S, K, r, σ, t, N)
    Δt = t / N
    u = exp(σ * √Δt)
    d = 1 / u
    R = exp(r * Δt)
    p = (R - d) / (u - d)
    q = 1 - p
    tree = zeros((N + 1, N + 1))
    for k = N:-1:0
        for j = 0:k
            tree[j+1, k+1] = max(0, K - S * exp((2 * j - k) * σ * √Δt)) # binomial tree
        end
    end
    Z = tree[:, N+1]
    for n = N-1:-1:0
        for i = 0:n
            #x = K - S * exp((2 * i - n) * σ * √Δt)
            x = tree[i+1, n+1]
            y = (q * Z[i+1] + p * Z[i+2]) / R
            Z[i+1] = max(x, y)
        end
    end

    return Z[1]
end
const N(x) = cdf(Normal(), x)

function d1(T, s, K, r, sigma)
    return 1 / (sigma * sqrt(T)) * (log(s / K) + (r + sigma^2 / 2) * T)
end

function d2(T, s, K, r, sigma)
    return d1(T, s, K, r, sigma) - sigma * sqrt(T)
end

function put_price(T, s, K, r, sigma)
    N(-d2(T, s, K, r, sigma)) * K * exp(-r * T) - N(-d1(T, s, K, r, sigma)) * s
end

function square_error_eu(N, T, s, K, r, sigma)
    y_0 = log(s)
    exact_price = put_price(T, s, K, r, sigma)
    err = zeros(N)
    for i = 1:N
        con_price = eu_put(s, K, r, sigma, T, i)
        err[i] = (con_price - exact_price)^2
    end
    return err
end



app_am = am_put(100, 95, 0.02, 0.1, 1, 200)
app_eu = eu_put(100, 95, 0.02, 0.1, 1, 200)
err = square_error_eu(400, 1, 100, 95, 0.02, 0.1)
coef = err[1]
n = 400
y_1 = coef * ones(n) ./ (1:n)
y_2 = coef * ones(n) ./ (1:n) .^ 2
y_3 = coef * ones(n) ./ (1:n) .^ 0.5


plot(1:n, [err, y_1, y_2, y_3], label = ["SquareError" "1/N" "1/N^2" "1/N^0.5"])
err_eu = square_error_eu(2000, 1, 100, 95, 0.02, 0.1)
plot(100:2000, err_eu[100:end], label = ["Euro-SquareError"])

Am_price = [am_put(100, 95, 0.02, 0.1, 1, n) for n = 20:2000]
#err_1 = square_error_am(2000, 1, 100, 95, 0.02, 0.1)
plot(1:1981, Am_price, label = ["AM-Price"])
#coef = err[50]
#err = err[50:end]
#coef_1 = err_1[1]
#n = 400
#y_4 = coef_1 * ones(n) ./ (1:n)
#y_5 = coef_1 * ones(n) ./ (1:n) .^ 2
#y_6 = coef_1 * ones(n) ./ (1:n) .^ 0.5

#plot(1:n, [err_1, y_4, y_5, y_6], label = ["SquareError" "1/N" "1/N^2" "1/N^0.5"])
