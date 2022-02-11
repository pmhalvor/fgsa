#!/bin/bash

# Job name:
#SBATCH --job-name=fgsa-test
# Project:
#SBATCH --account=ec37
# Wall time limit: --time=DD-HH:MM:SS
#SBATCH --time=00:10:00
# Memory requested:
#SBATCH --mem-per-cpu=2G
#SBATCH --qos=devel
# Run on GPU
#SBATCH --partition=accel
#SBATCH --gpus=1

## Set up job environment:
set -o errexit  # Exit the script on any error
set -o nounset  # Treat any unset variables as an error

# modules
module --quiet purge  # Reset the modules to the system default
module load Miniconda3/4.9.2
module load PyTorch/1.7.1-foss-2020b


## Call test folder:
cd code
pytest . -vv