#!/bin/bash
#SBATCH -t 5:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node 48
#SBATCH --mem=50GB
#SBATCH -o job.out
#SBATCH -e job.err
#SBATCH -p g100_usr_prod
#SBATCH -A tra24_qc_week_0

python exact_solver.py
