using Random
Random.seed!(0)
using Base.MathConstants
using Statistics
using Measurements
using LinearAlgebra

# source(s) of code used = Kochenderfer and Wheeler algorithm 5.4

abstract type DescentMethod end
mutable struct NesterovMomentum <: DescentMethod 
    alpha # learning rate
    beta # momentum decay
    v # momentum
end

function init!(M::NesterovMomentum, x, df, f)
    M.v = zeros(length(x))
    return M 
end

function step!(M::NesterovMomentum, x, df, f) 
    alpha, beta, v = M.alpha, M.beta, M.v
    v[:] = beta.*v .- alpha.*df(x + beta*v)
    return x + v
end


```
Tolerances and stopping conditions are hard coded in this function. Change as needed.
```
function nesterov_descent(x0, df, f)
    x_tol = 0.0005
    f_tol = 0.01
    
    NM = NesterovMomentum(0.0005, 0.9, 0) # tune alpha and beta params
    M = init!(NM, x0, df, f)
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
        
        if counter >= 500 # hard limit on number of iterations
            break
        end
        
        cur_x = next_x
        cur_y = next_y
    end
    
    return next_x, next_y#, cur_y, time, counter #
end

# ————————— TESTING BELOW ————————— #
```
Run the file to test. Be sure to uncomment lines 38, 39, 48, 49, and 64.
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

x, conv, step, time = get_stats(nesterov_descent, x_generated, d_rosenbrock, rosenbrock)
print("Rosenbrock\n")
print("Estimated x\tConvergence error\tSteps taken\tTime taken\n")
print(x, "\t", conv, "\t\t", step, "\t", time, "\n")

x, conv, step, time = get_stats(nesterov_descent, x_generated, d_custom_func, custom_func)
print("Custom Function\n")
print("Estimated x\tConvergence error\tSteps taken\tTime taken\n")
print(x, "\t", conv, "\t\t", step, "\t", time, "\n")