# tiktak_sample_julia
Sample code for running the TikTak algorithm (https://www.fatihguvenen.com/tiktak). The code is specifically designed to be run on a HPC that uses a SLURM workload manager.

Files 
- ```tiktak_base.jl```: base code which contains all of the necessary functions to implement the TikTak algorithm
- ```tiktak_solve.jl```: A wrapper to solve for the global minimum via TikTak
- ```run_cluster.jl```: Sample code file to implement TikTak on a simple function (Rosenbrock)
- ```RUN_algorithm.slurm```: Sample SLURM script to submit job to a SLURM workload manager on a HPC (to be used on linux)
- ```submit_to_cluster.sh```: shell script to submit RUN_algorithm.slurm

```run_cluster.jl``` 
