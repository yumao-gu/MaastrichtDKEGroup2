#!/bin/bash

#SBATCH --job-name=normal_ci_sample100000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=56
#SBATCH --mem-per-cpu=2GB
#SBATCH --time=24:00:00
#SBATCH --account=um_dke

python3 CI.py
