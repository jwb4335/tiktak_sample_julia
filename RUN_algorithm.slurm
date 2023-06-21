#!/bin/bash
sbatch <<EOT
#!/bin/bash

## SOLVE THE TIKTAK ALGORITHM ON THE SLURM CLUSTER

#SBATCH -N 20									# Nodes
#SBATCH -n 300									# Tasks 
#SBATCH -c 1                            	  	# cores per task
#SBATCH -t 168:00:00							# time limit
#SBATCH --mem-per-cpu=3G						# 2G seems to work well
#SBATCH -p common-old                           # Partition
# #SBATCH --mail-type=ALL                         # Mail alerts?
# #SBATCH --mail-user=jwb74@duke.edu              # Email to mail to
#SBATCH -o ""$1".txt"                           # output filename
#SBATCH -e ""$1".err"                           # error filename
#SBATCH -J $2
									# Job name
# Run Julia script
julia tiktak_wrapper.jl 

EOT
