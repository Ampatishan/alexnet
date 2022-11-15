#!/bin/bash 
#SBATCH --account=def-punithak
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --mem=64G
#SBATCH --time=40:00:00
#SBATCH --mail-user=ampatish@ualberta.ca
#SBATCH --mail-type=ALL

source alexnet/bin/activate
python main.py