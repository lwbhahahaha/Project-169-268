using Random
Random.seed!(0)
using Base.MathConstants
using Statistics
using Measurements
using LinearAlgebra

# source(s) of code used = Kochenderfer and Wheeler algorithm 5.7
abstract type DescentMethod end
mutable struct Adadelta <: DescentMethod 
    gamma_s # gradient decay
    gamma_x # update decay
    epsilon # small value
    s # sum of squared gradients
    u # sum of squared updates 
end
function init!(M::Adadelta, x, df, f) 
    M.s = zeros(length(x))
    M.u = zeros(length(x))
    return M
end
function step!(M::Adadelta, x, df, f)
    gamma_s, gamma_x, epsilon, s, u, g = M.gamma_s, M.gamma_x, M.epsilon, M.s, M.u, df(x) 
    s[:] = gamma_s*s + (1-gamma_s)*g.*g
    delta_x = - (sqrt.(u) .+ epsilon) ./ (sqrt.(s) .+ epsilon) .* g
    u[:] = gamma_x*u + (1-gamma_x)*delta_x.*delta_x
    return x + delta_x 
end


```
Initialization parameters and stopping conditions are hard coded in this function. Change as needed.
```
function adadelta_descent(x0, df, f)
    x_tol = 0.0005
    f_tol = 0.01
    
    AD = Adadelta(0.95, 0.95, 1e-3, 0, 0) # tune decay params
    M = init!(AD, x0, df, f)
    #time = 0 #
    #counter = 0 #
    
    cur_x = x0
    cur_y = f(x0)
    next_x = 0
    next_y = 0
    while true
        next_x, step_time = @timed step!(M, cur_x, df, f)
        next_y = f(next_x)
        #time += step_time #
        #counter += 1 #
                
        # stopping conditions
        if sqrt(sum((next_x .- cur_x).^2)) < x_tol || sqrt(sum((next_y .- cur_y).^2)) < f_tol
            break
        end
        
        if counter >= 1000 # hard limit on number of iterations
            break
        end
        
        cur_x = next_x
        cur_y = next_y
    end
    
    return next_x, next_y#, cur_y, time, counter #
end

# ————————— TESTING BELOW ————————— #
```
Run the file to test. Be sure to uncomment lines 40, 41, 50, 51, and 66.
```

# random values generated in 100x10 array ranging from (-10,10)
y_values = (rand(1000) .- 0.5) .* 20
x_signage = rand((-1,1), 1000)
x_values = y_values .* x_signage
x_generated = reshape(x_values, (100, 10))

function rosenbrock(x; a=1, b=5)
    sum = 0
    for i in 1:9
        sum += (a - x[i])^2 + b * (x[i+1] - x[i]^2)^2
    end
    return sum
end

function d_rosenbrock(x; a=1, b=5)
    dr = []
    for i in 1:9
        value = -2(a - x[i]) - 2b * (x[i+1] - x[i]^2) * 2x[i] + 2b * (x[i+1] - x[i]^2)
        append!(dr, value)
    end
    value = -2(a - x[10]) - 2b * (x[1] - x[10]^2) * 2x[10] + 2b * (x[1] - x[10]^2)
    append!(dr, value)
    return dr
end

custom_func(x) = sum(x.^4 .+ 2x.^3 .- x.^2 .+ 3x .+ 20)
d_custom_func(x) = 4x.^3 .+ 6x.^2 .- 2x .+ 3

function get_stats(descent_func, x0, df, f)
    x_vals, convergences, steps, times = [], [], [], []
    
    for i = 1:100
        x, y, prev_y, time, counter = descent_func(x0[i,:], df, f)
        append!(x_vals, x)
        append!(convergences, abs(y - prev_y))
        append!(steps, counter)
        append!(times, time)
    end
    
    x_stats = measurement(mean(x_vals), std(x_vals, corrected=true))
    conv_stats = measurement(mean(convergences), std(convergences, corrected=true))
    step_stats = measurement(mean(steps), std(steps, corrected=true))
    time_stats = measurement(mean(times), std(times, corrected=true))
    return x_stats, conv_stats, step_stats, time_stats
end

x, conv, step, time = get_stats(adadelta_descent, x_generated, d_rosenbrock, rosenbrock)
print("Rosenbrock\n")
print("Estimated x\tConvergence error\tSteps taken\tTime taken\n")
print(x, "\t", conv, "\t\t", step, "\t", time, "\n")

x, conv, step, time = get_stats(adadelta_descent, x_generated, d_custom_func, custom_func)
print("Custom Function\n")
print("Estimated x\tConvergence error\tSteps taken\tTime taken\n")
print(x, "\t", conv, "\t\t", step, "\t", time, "\n")