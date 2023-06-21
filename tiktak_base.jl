"""
Baseline code to solve for the global minimum of a non-differentiable objective using
a variant of the TikTak algorithm
"""

using ArgCheck: @argcheck
using Base.Threads: @spawn, fetch
using Parameters: @unpack
using Sobol: SobolSeq, Sobol
using NLopt


#==============================================================
MINIMIZATION ROUTINE SETTINGS FROM NLOPT. NOT TIKTAK YET
==============================================================#

struct MinimizationProblem{F,T<:AbstractVector}
    # The function to be minimized
    objective::F
    # Lower bounds (a vector of real numbers)
    lower_bounds::T
    # Upper bounds (a vector of real numbers)
    upper_bounds::T
    """
    
    Define a minimization problem.

    - `objective` is a function that maps `ℝᴺ` vectors to real numbers. It should accept all
       `AbstractVector`s of length `N`.
    - `lower_bounds` and `upper_bounds` define the bounds, and their lengths implicitly
      define the dimension `N`.

    The fields of the result should be accessible with the names above.
    """
    function MinimizationProblem(objective::F, lower_bounds::T, upper_bounds::T) where {F,T}
        @argcheck all(lower_bounds .< upper_bounds)
        new{F,T}(objective, lower_bounds, upper_bounds)
    end
end

"""
What constitutes optimization success? Note, this object will be return with each local minimization run
Usually, maxeval will be reached
"""
NLopt_ret_success = Set([:SUCCESS, :STOPVAL_REACHED, :FTOL_REACHED, :XTOL_REACHED,
                               :MAXEVAL_REACHED, :MAXTIME_REACHED])

"""
Baseline local method options. Can be changed for each problem
"""
Base.@kwdef struct NLoptLocalMethod{S}
    algorithm::NLopt.Algorithm
    xtol_abs::Float64 = 1e-8
    xtol_rel::Float64 = 1e-8
    maxeval::Int = 0
    maxtime::Float64 = 0.0
    # Return values which are considered as “success”.
    ret_success::S = NLopt_ret_success
end

"""
A wrapper for algorithms supported by `NLopt`. Used to construct the corresponding
optimization problem. All return values in `ret_success` are considered valid (`ret` is also
kept in the result), all negative return values are considered invalid.

See the NLopt documentation for the options. Defaults are changed slightly.
"""

function NLoptLocalMethod(algorithm::NLopt.Algorithm; options...)
    NLoptLocalMethod(; algorithm = algorithm, options...)
end

"""
Solve `minimization_problem` using `local_method`, starting from `x`. Return a
`LocationValue`
"""
function local_minimization(local_method::NLoptLocalMethod,
                            minimization_problem::MinimizationProblem, x)
    @unpack algorithm, xtol_abs, xtol_rel, maxeval, maxtime, ret_success = local_method
    @unpack objective, lower_bounds, upper_bounds = minimization_problem
    opt = NLopt.Opt(algorithm, length(x))
    opt.lower_bounds = lower_bounds
    opt.upper_bounds = upper_bounds

    # If a method `objective(x, grad)` exists, use it; otherwise assume objective is not
    # differentiable.
    opt.min_objective = applicable(objective, x, x) ? objective : nlopt_nondifferentiable_wrapper(objective)
    opt.xtol_abs = xtol_abs
    opt.xtol_rel = xtol_rel
    opt.maxeval = maxeval
    opt.maxtime = maxtime
    optf, optx, ret = NLopt.optimize(opt, x)
    ret ∈ ret_success ? (value = optf, location = optx, ret = ret) : nothing
end
"""
If using a non-differentiable function (does not have a gradient), 
add an empty gradient as NLopt requires a gradient to be defined
"""
function nlopt_nondifferentiable_wrapper(fn)
    function f̃(x,grad)              # wrapper for NLopt
        @argcheck isempty(grad)     # ensure no derivatives are asked for
        return fn(x)
    end
    return f̃
end

#==============================================================
TIKTAK ALGORITHM
==============================================================#
"""
A TikTak object! These numbers will all be determined by quasirandom_N
"""
struct TikTak
    quasirandom_N::Int
    initial_N::Int
    θ_min::Float64
    θ_max::Float64
    θ_pow::Float64
end

function TikTak(quasirandom_N; keep_ratio = 0.1, θ_min = 0.1, θ_max = 0.995, θ_pow = 0.5)
    @argcheck 0 < keep_ratio ≤ 1
    TikTak(quasirandom_N, ceil(keep_ratio * quasirandom_N), θ_min, θ_max, θ_pow)
end

"""
Helper: How convex combination weight param changes by iteration
Directly from original paper
"""
function _weight_parameter(t::TikTak, i)
    @unpack initial_N, θ_min, θ_max, θ_pow = t
    clamp((i / initial_N)^θ_pow, θ_min, θ_max) * Float64(i>0) + Float64(i == 0)
end

"""
Helper: If function returns NaN, return a really large number
"""
function _acceptable_value(value)
    if isnan(value)
        return 1e10
    else
        return value
    end
end

            
"""
Helper: Evaluate objective at an initial location (NOT a minimization routine)
Used in first part of algorithm to find "promising" random points
"""
function _objective_at_location(objective, location)
    value = objective(location)
    value = _acceptable_value(value)
    (; location, value)
end


"""
QUASIRANDOM POINTS FUNCTION TO COMPUTE INITIAL LOCATION, 
COULD BE STREAMLINED AND USED BUT NOT USED IN CURRENT CODE
"""
function sobol_starting_points(minimization_problem::MinimizationProblem, N::Integer,
    use_threads::Bool)
    @unpack objective, lower_bounds, upper_bounds = minimization_problem
    s = SobolSeq(lower_bounds, upper_bounds)
    skip(s, N)                  # better uniformity
    points = Iterators.take(s, N)
    _initial(x) = _objective_at_location(objective, x)

    if use_threads
       map(fetch, map(x -> @spawn(_initial(x)), points))
    else
       map(_initial, points)
    end
end


"""
Function to get a sequence of Sobol (quasirandom points)
"""
function get_starting_sobol_points(minimization_problem::MinimizationProblem, N::Integer)
    @unpack objective, lower_bounds, upper_bounds = minimization_problem
    s = SobolSeq(lower_bounds, upper_bounds)
    skip(s, N)                  # better uniformity
    points = Iterators.take(s, N)
    return points
end

""" 
Helper: Keep points with lowest objective function values
"""
function _keep_lowest(xs, N)
    @argcheck 1 ≤ N ≤ length(xs)
    partialsort(xs, 1:N, by = p -> p.value)
end

function _solve_local(initial_point,i,visited_minimum)
    """
    Helper: solve a local minimization problem
    inputs:
        initial point: candidate location
        i: initial point's location in sequence of possible points for tiktak
        visited_minimum: current minimum
    
    returns local minimum of initial point
    """

    # @printf "current minimum: %.4f" visited_minimum.value
    θ = _weight_parameter(multistart_method, i)
    x = @. (1 - θ) * initial_point.location + θ * visited_minimum.location
    local_minimum = local_minimization(local_method, minimization_problem, x)

    return local_minimum
end

function _solve_local(initial_point,i,visited_minimum)
    """
    Helper: solve a local minimization problem
    inputs:
        initial point: candidate location
        i: initial point's location in sequence of possible points for tiktak
        visited_minimum: current minimum
    
    returns local minimum of initial point
    """
    x = @. (1 - θ) * initial_point.location + θ * visited_minimum.location
    local_minimum = local_minimization(local_method, minimization_problem, x)
    
    value = local_minimum.value
    location = local_minimum.location
    ret = local_minimum.ret


    return (; value,location, ret )
end


function _solve_local_no_weight(initial_point,minimization_problem,local_method)
    """
    Helper: solve a local minimization problem
    inputs:
        initial point: candidate location
        minimization_problem: objective
        local_method: local minimizer (e.g. neldermead)
    
    returns local minimum of initial point
    """
    # @printf "current minimum: %.4f" visited_minimum.value
    # θ = _weight_parameter(multistart_method, i)
    x = initial_point.location
    local_minimum = local_minimization(local_method, minimization_problem, x)

    value = local_minimum.value
    location = local_minimum.location
    ret = local_minimum.ret


    return (; value,location, ret )
end


function tiktak_finisher(multistart_method::TikTak,
    minimization_problem,
    local_method,
    local_minima;
    start_point = 1)
    """
    The "TikTak" step of TikTak

    This function is designed to be run on a local machine (i.e., one node)
    it iterates forward through the possible points, increasing the θ parameter
    for each iteration, finding a new convex combination for a candidate minimum
    and using local minimization on each point

    returns the "global" minimum 
    """
    # Redefine the local method, might want to increase max_eval
 

    @unpack quasirandom_N, initial_N, θ_min, θ_max, θ_pow = multistart_method
    @unpack objective, lower_bounds, upper_bounds = minimization_problem

    total_length = length(local_minima)
    function _step(visited_minimum, (i, initial_point))
        if i == 1
            f = visited_minimum.value
            println("current minimum: $f")
        end
        flush(stdout)
        println("iteration $i of $total_length")

        θ = _weight_parameter(multistart_method, i)
        x = @. (1 - θ) * initial_point.location + θ * visited_minimum.location
        local_minimum = local_minimization(local_method, minimization_problem, x)
        local_minimum ≡ nothing && return visited_minimum
        if local_minimum.value< visited_minimum.value
            f = local_minimum.value
            println("New minimum reached, obj fn val: $f")

            println(round.(local_minimum.location';digits=4))
        else
            f = visited_minimum.value
            println("Staying at current minimum, obj fn val: $f")
            println(round.(visited_minimum.location';digits=4))

        end


        local_minimum.value < visited_minimum.value ? local_minimum : visited_minimum

    end

    sort!(local_minima, by = p -> p.value)

    foldl(_step, collect(enumerate(Iterators.drop(local_minima, 1)))[start_point:end]; init = first(local_minima))
end


function _update_main(i,visited_minimum,(j,initial_point); multistart_method = TikTak_object)
    """ 
    Helper: Get the convex combinations of the current minimum with all other candidate minima
    Inputs:
        i: index of current minimum, need this to update weight param θ
        visited_minimum: current minimum
        (j,initial_point): j needed to give index of candidate minimum, initial_point, the candidate minimum itself
    """
    θ = _weight_parameter(multistart_method, j)
    (idx = j+i,location = @. (1 - θ) * initial_point.location + θ * visited_minimum.location)
end

function tiktak_finisher_cluster(TikTak_object::TikTak,
                                minimization_problem,
                                local_method, 
                                local_minima)
    """
    Utilize a cluster to engage in TikTak
    Idea: 
        Get the convex combinations with current minimum and all lesser points as in TikTak
        Solve for the local minima of all candidate points
        Update to the new minimum (if it exists)
        Continue until we get a minimum
    NOTES:
        - TIME: In the worst-case, it takes as long as the standard TikTak, however, it's possible that you can skip a lot
        of steps by computing all possible local minima at once
        - SPACE: Very computationally inefficient, should only be implemented on a cluster
        - If you want to run things locally, just use the function "tiktak_finisher" above
    """
    # Unpack tiktak params
    @unpack quasirandom_N, initial_N, θ_min, θ_max, θ_pow = TikTak_object;

    # Get the number of iterations that shall be done
    n_iter = length(local_minima)

    # Start the iterative procedure
    i = 1

    # Set the best local minimum so far.
    visited_minimum = first(local_minima)


    #===================================================================================
    While loop: calculate local minimum for each possible candidate, given iteration i
    Once done, update to new local minimum and repeat
    Given cluster capabilities, can salve for n_iter local minima quickly
    After that, we reduce by at least one iteration, and at most 100.
    ===================================================================================#
    while i < n_iter
        if i == 1
            f = visited_minimum.value
            println("at first iter, current minimum: $f")
            flush(stdout)
        # else
        #     curr_val = visited_minimum.value
        #     println("Going to iter $i, current minimum: $curr_val")
        #     println(visited_minimum.location)
        end



        # check = pmap(_get_min, [visited_minimum.location])
        # sanity_check = check.value
        # println("is the minimum doing what it says it is? $sanity_check")
        # flush(stdout)



        # Calculate the candidate points as in Tiktak.
        _update((j,initial_point)) = _update_main(i,visited_minimum,(j,initial_point))

        # Map the fn _update so it returns a collection of points
        points = map(_update, enumerate(Iterators.drop(local_minima, i)))

        #===================================================================================
        HERE is the big step 
        ===================================================================================#

        # Function to run local minimum on all points
        @everywhere _get_min(x) = local_minimization(local_method, minimization_problem, x)

        # Get the collection of all possible locations
        X = [z.location for z in points]

        # SOLVE FOR LOCAL MINIMA ON CLUSTER
        new_minima = pmap(_get_min,X)

        filter!(x -> !isnothing(x), new_minima)
        
        # Get the objective function values for each new minimum
        values = [x.value for x in new_minima]

        
        # CHECK: do any of the new points beat the current minimum
        f = findfirst(values .< visited_minimum.value) 

        # Get the argmin of the values (to be used below)
        # NOTE: This is different to original TikTak. In TiKtak, you go to the first minimum that beats the current, 
        # I go to the BEST. Not clear which is better
        ff = argmin(values)

        #===================================================================================
        If we find a new minimum, ff gives its index, so reset i += ff, so we go to the 
        new best minimum
        Remember, ff is from the index of the reduced vector of candidate minima 
        Otherwise, if the current iter is the true minimum, then f = nothing
        and we are done 
        ===================================================================================#
        if isnothing(f)
            println("Minimization is done: global minimum index: $i")
            curr_val = visited_minimum.value
            println("global minimum value $curr_val, global minimum:")
            display(visited_minimum.location)
            i = 1e10;
        else
            i += ff
            visited_minimum = new_minima[ff]
            println("Updating new minimum to $i and re-running recursive minimization")
            curr_val = visited_minimum.value
            println("current minimum value $curr_val, current minimum:")
            display(visited_minimum.location)
        end

    end

    return visited_minimum

end
