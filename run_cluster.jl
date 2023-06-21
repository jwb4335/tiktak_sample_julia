""" 
PRELIMINARIES
load packages, set workers etc. 
"""

println("Instantiating, pre-compiling...")
# I have an environment called "env_cluster," you will have to change
# Must instantiate and precompile first (but only do this on the main node to avoid race conditions)
# Bare minimum of packages needed: ArgCheck, Parameters, Sobol, NLopt, JLD
using Pkg; Pkg.activate("env_cluster"); Pkg.instantiate(); Pkg.precompile();
sleep(10)
# These are the cluster packages I use
using Distributed, SlurmClusterManager; 

println("initializing cluster and adding slurm workers")
flush(stdout)
# Add your nodes (uncomment this and comment out code below if you are running on a cluster)
# addprocs(SlurmManager());

# Uncomment this if you want to experiment on local machine
NPROCS = 5;
if nprocs()<NPROCS
   addprocs(NPROCS - nprocs());
end

println("activating environment on all nodes")
flush(stdout)
@everywhere begin 
    using Pkg; Pkg.activate("env_cluster"); 

    # THESE ARE THE PACKAGES I NEED, BUT THE ABSOLUTELY NEEDED PACKAGES FOR TIKTAK ARE LOADED IN FILE tiktak_base.jl
    # using Distributed, Distributions, Random, Statistics, Parameters, DataFrames, DataFramesMeta, Interpolations,
    # JLD, Base.Threads, FixedEffectModels, NelderMead, NLopt, LinearAlgebra, Printf, FastGaussQuadrature, StatsBase, 
    # CategoricalArrays, Missings, Kronecker, Tullio, FiniteDiff, CSV
    
    include("tiktak_base.jl");

    include("tiktak_solve.jl");

    # Include everywhere any files you may need, such as your VFI function, and any helper functions you need
    # include("UpdateVar.jl");
    # include("SetupWageGame.jl");
    # include("WageGame.jl");
    # include("utils.jl");
    # include("SetupParams.jl");
    # include("Initialize.jl");
    # include("SolverStatic.jl");
    # include("SimulateSlim.jl");
    # include("Moments.jl");
    # include("ParsCheck.jl");
    # include("ValueFunctionIter.jl");
    # include("MinProblemDef.jl");
    # include("InputCreator.jl");
    
end
println("INITIATED!")
flush(stdout)


##
"""
Solution step!
"""
# Define the TikTak object

# How many initial quasirandom points?
N_tiktak = 10000; # I usually go high, like 100k, but set it low for the example
# What percentage to keep? 
keep_ratio = 0.5; # I would usually keep about 500 points

TikTak_object =  TikTak(N_tiktak; keep_ratio = keep_ratio);

# How many initial local evaluations?
local_maxeval_initial = 100;

# How many local evaluations in the TikTak step?
local_maxeval_final = 100;

# Silly sample function: Rosenbrock, stolen from Optim page 
# http://julianlsolvers.github.io/Optim.jl/v0.9.3/user/minimization/#minimizing-a-multivariate-function
# Minimized at [1., 1.]

@everywhere rosenbrock(x::Vector{Float64}) = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2;
lower = [0., 0.];
upper = [100., 100.];

quasirandom_points, all_points, local_minima, global_minimum = 
TikTak_SOLVE(TikTak_object,rosenbrock, lower, upper; 
    local_maxeval_initial = local_maxeval_initial,
    local_maxeval_final = local_maxeval_final);

minimum = global_minimum.location;
obj_fn_val = global_minimum.value;
println("done! global_min = $minimum; value = $obj_fn_val");