# tiktak_sample_julia
Sample code for running the TikTak algorithm (https://www.fatihguvenen.com/tiktak). The code is specifically designed to be run on a HPC that uses a SLURM workload manager.

Files 
- ```tiktak_base.jl```: base code which contains all of the necessary functions to implement the TikTak algorithm
- ```tiktak_solve.jl```: A wrapper to solve for the global minimum via TikTak
- ```run_cluster.jl```: Sample code file to implement TikTak on a simple function (Rosenbrock)
- ```RUN_algorithm.slurm```: Sample SLURM script to submit job to a SLURM workload manager on a HPC (to be used on linux)
- ```submit_to_cluster.sh```: shell script to submit RUN_algorithm.slurm

```run_cluster.jl``` will currently run locally, but
- you should load in your own environment on lines https://github.com/jwb4335/tiktak_sample_julia/blob/4319d6bd385d33b6807459829e16091412d45e39/run_cluster.jl#L10 and https://github.com/jwb4335/tiktak_sample_julia/blob/4319d6bd385d33b6807459829e16091412d45e39/run_cluster.jl?plain=1#L29
- and make sure you have these packages added: Distributed, SlurmClusterManager, ArgCheck, Parameters, Sobol, NLopt, JLD

 If you want to use ```run_cluster.jl``` on a cluster, 
 - comment out https://github.com/jwb4335/tiktak_sample_julia/blob/4319d6bd385d33b6807459829e16091412d45e39/run_cluster.jl#L21-24
 - and uncomment https://github.com/jwb4335/tiktak_sample_julia/blob/4319d6bd385d33b6807459829e16091412d45e39/run_cluster.jl#L18
