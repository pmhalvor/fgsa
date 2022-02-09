#!/bin/bash

# Job name:
#SBATCH --job-name=fgsa
# Project:
#SBATCH --account=ec37
# Wall time limit: --time=DD-HH:MM:SS
#SBATCH --time=00:30:00
# Memory large bc BertTokenizer used:
#SBATCH --mem 20G


## Set up job environment:
set -o errexit  # Exit the script on any error
set -o nounset  # Treat any unset variables as an error


# modules
module --quiet purge  # Reset the modules to the system default
module load Miniconda3/4.9.2
module load PyTorch/1.7.1-foss-2020b


## Do some work:
cd ../code
python preprocess.py --interactive 0 --overwrite 1