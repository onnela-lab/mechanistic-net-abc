#!/bin/bash
#SBATCH -J Rec50	# A single job name for the array
#SBATCH -n 7                # Number of cores
#SBATCH -N 1                # Ensure that all cores are on one machine
#SBATCH -t 7-00:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p hsph,shared      # Partition to submit to
#SBATCH --mem=8000          # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o contact_network_50nodes_SMC_sel_summaries_obs3_recursive_PA_RA_TF_obs_50000sim_%j.out    # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e contact_network_50nodes_SMC_sel_summaries_obs3_recursive_PA_RA_TF_obs_50000sim_%j.err     # File to which STDERR will be written, %j inserts jobid

module load Anaconda3/2020.11
source activate mechanistic_env

python3 contact_network_50nodes_SMC_sel_summaries_obs3_recursive_PA_RA_TF_obs_50000sim.py ${SLURM_ARRAY_TASK_ID}