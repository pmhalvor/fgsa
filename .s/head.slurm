#!/bin/bash

# Job name:
#SBATCH --job-name=fgsa-head
# Project:
#SBATCH --account=ec37
# Wall time limit: --time=DD-HH:MM:SS
#SBATCH --time=10:30:00
# Memory requested:
#SBATCH --mem-per-cpu=30G
# Run on GPU
#SBATCH --partition=accel
#SBATCH --gpus=1

## Set up job environment:
set -o errexit  # Exit the script on any error
set -o nounset  # Treat any unset variables as an error

# modules
module --quiet purge  # Reset the modules to the system default
module use -a /fp/projects01/ec30/software/easybuild/modules/all/
module load nlpl-pytorch/1.7.1-foss-2019b-cuda-11.1.1-Python-3.7.4
module load nlpl-scikit-bundle/0.22.2.post1-foss-2019b-Python-3.7.4
module load nlpl-tokenizers/0.10.2-foss-2019b-Python-3.7.4


## Do some work:
cd $HOME/nlp/msc/fgsa/src
python study.py $1 
