"""
Global solver via TikTak. Intended to be run via distributed computing. E.g., combined with ClusterManager.jl or SlurmClusterManager.jl

Inputs: 
- TikTak_object: A pre-defined TikTak object
- objective_function: The objective function for which you want a global minimum.  
  MUST only take parameters as inputs, for example you can define an anonymous function in your wrapper,
  which will inherit other important non-estimated params from a pre-defined function
- lower: Lower bound on parameters
- upper: upper bound on parameters

Outputs: 
- quasirandom_points: #N_tiktak points, along with initial function evaluations. Intended to identify "promising" starting points
- all_points: best #ceil(N_tiktak*keep_ratio) quasirandom points, plus prepended points
- local_minima: Local minima of all_points
- global_minimum: outcome of TikTak algorithm
"""

using JLD # Needed if saving intermediate points

function TikTak_SOLVE(
    TikTak_object::TikTak,
    # Solution params
    objective_function::Function, # Defined objective function. MUST be just a function of the input parameters, i.e. define an anonymous function in your wrapper script
    lower::Vector{Float64}, # Lower bounds on parameters 
    upper::Vector{Float64}, # Upper bounds on Parameters
    ;
    # Other stuff 
    path_to_save=nothing, # Path to save intermediate steps, defaults to nothing which means no save
    local_maxeval_initial = 200, # How many evaluations for local minimizer in the initial local step
    local_maxeval_final = 400, # How many evaluations for local minimizer in the TikTak (final) step
    prepend_points = nothing, # Do you have promising points already? A vector of promising paremeter vectors, should have type Vector{Vector{Float64}}
    local_minimization_algorithm = NLopt.LN_NELDERMEAD # What local algorithm to use? Must be from NLopt. Defaults to LN_NELDERMEAD, but also LN_SBPLX can work
    )
    

    # @unpack_namedtuple sim_params

    # Define the minimization problem!!!!
    local_method = NLoptLocalMethod(local_minimization_algorithm; maxeval = local_maxeval_initial);
    minimization_problem = MinimizationProblem(objective_function,lower,upper);    
    # Need to define the minimization problem and necessary parameters everywhere. You will have to modify this yourself
    @everywhere begin 
        local_method = NLoptLocalMethod($local_minimization_algorithm; maxeval = $local_maxeval_initial);
        minimization_problem = MinimizationProblem($objective_function,$lower,$upper);    
    end   

    """
    START: TikTak 
    set the TikTak settings
    """

    @unpack quasirandom_N, initial_N, θ_min, θ_max, θ_pow = TikTak_object;

    # Get the sobol points
    points = get_starting_sobol_points(minimization_problem,N_tiktak);


    #===========================================================================
    ############################################################################
    Big step here: SOLVING FOR N INITIAL SOBOL POINTS

    pmap function sends each point to a different worker

    ############################################################################
    ===========================================================================#
    # Define helper function to solve for the obj. fn values at the initial locations
    _initial = x -> _objective_at_location(minimization_problem.objective, x)

    println("solving for initial locations...")
    flush(stdout)
        quasirandom_points = pmap(_initial,points)
    flush(stdout)
    
    # Save quasirandom points?
    if ~isnothing(path_to_save)
        # Take a snapshot of the quasirandom points
        filename_quasirandom =  joinpath(path_to_save,"quasirandom_points.jld")
        @save filename_quasirandom quasirandom_points
    end

    # Keep the top proportion of points given the keep_ratio parameter
    initial_points = _keep_lowest(quasirandom_points, initial_N);

    # Define helper function to solve for the local minima of promising points
    find_local_minima = initial_point -> _solve_local_no_weight(initial_point,minimization_problem,local_method)

    # concatenate the prepend_points with the estimated points
    if ~isnothing(prepend_points)
        
        # Re-run locally on prepend points if wanted, may make sense if matching different moments
        println("Re-running on prepend")
        points_in = [x.location for x in prepend_points]

        #===========================================================================
        ############################################################################
        Another pmap step
        ############################################################################
        ===========================================================================#
        prepend_points = pmap(_initial,points_in)

        # Collect all points together 
        all_points = collect(vcat(prepend_points,
        convert(Vector{NamedTuple}, initial_points)))
    else
        all_points = collect(initial_points)
    end



    # Sometimes it thinks all_points has mixed types, just re-define it
    all_points = [(value = x.value, location = x.location) for x in all_points]
    
    # Sort points by objective function value
    sort!(all_points, by = p -> p.value)

    # Save all of your points so far
    if ~isnothing(path_to_save)
        # Save promising points!!
        file_all_points = joinpath(path_to_save,"all_points.jld")
        @save file_all_points all_points
        # all_points = load(file_all_points)["all_points"]
    end

    
    visited_minimum = first(all_points)
    f = visited_minimum.value
    println("current minimum: $f")

    #===========================================================================
    ############################################################################
    VERY big step here: FINDING LOCAL MINIMA OF ~ceil(N*0.1) PROMISING STARTING POINTS

    pmap function sends each point to a different worker

    ############################################################################
    ===========================================================================#
    println("Solving for local minima of promising points...")
    flush(stdout)
    local_minima = pmap(find_local_minima,all_points)
    flush(stdout)
    
    sort!(local_minima, by = p ->p.value);

    if  ~isnothing(path_to_save)
        filename_local_minima =  joinpath(path_to_save,"local_minima.jld")
        @save filename_local_minima local_minima
    end

    local_method_final = NLoptLocalMethod(NLopt.LN_NELDERMEAD; maxeval = local_maxeval_final); # Also increase the number of iters

    global_minimum = tiktak_finisher_cluster(TikTak_object,minimization_problem,local_method_final,local_minima)

    if  ~isnothing(path_to_save)
        filename_global_minimum =  joinpath(path_to_save,"global_minimum.jld")
        @save filename_global_minimum global_minimum
    end

    return quasirandom_points, all_points, local_minima, global_minimum;

end;

