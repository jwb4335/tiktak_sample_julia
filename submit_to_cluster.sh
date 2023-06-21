
## slurm file takes two arguments: name of results file, name of job.
#!/bin/bash
sbatch --output=/dev/null RUN_algorithm.slurm  "res_tiktak_full" "tiktak_full"; 
