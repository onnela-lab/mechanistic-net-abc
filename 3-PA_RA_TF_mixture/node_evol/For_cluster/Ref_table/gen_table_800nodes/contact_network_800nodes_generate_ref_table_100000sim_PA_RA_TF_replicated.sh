#!/bin/bash
#SBATCH -J 800nodes	# A single job name for the array
#SBATCH -n 1                # Number of cores
#SBATCH -N 1                # Ensure that all cores are on one machine
#SBATCH -t 7-00:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p hsph,shared      # Partition to submit to
#SBATCH --mem=8000          # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o contact_network_800nodes_generate_ref_table_100000sim_PA_RA_TF_replicated_%j_%a.out    # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e contact_network_800nodes_generate_ref_table_100000sim_PA_RA_TF_replicated_%j_%a.err     # File to which STDERR will be written, %j inserts jobid

module load Anaconda3/2020.11
source activate mechanistic_env

python3 contact_network_800nodes_generate_ref_table_100000sim_PA_RA_TF_replicated.py ${SLURM_ARRAY_TASK_ID}