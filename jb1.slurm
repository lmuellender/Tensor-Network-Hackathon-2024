#!/bin/bash
#SBATCH -t 1:00:00
#SBATCH -N 2
#SBATCH --ntasks-per-node-16
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:4
#SBATCH --mem=10GB
#SBATCH -o job.out
#SBATCH -e job.err
#SBATCH -p boost_usr_prod
#SBATCH -A tra24_qc_week

module load openmpi

export OMP PROC_BIND=true
mpirun -n 32/cop.py