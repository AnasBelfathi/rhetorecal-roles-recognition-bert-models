#!/bin/bash
#SBATCH --job-name=bert_rr_ft              # nom du job
#SBATCH --output=./job_out_err/bert_rr_ft_%j.out          # nom du fichier de sortie
#SBATCH --error=./job_out_err/bert_rr_ft_%j.err           # nom du fichier d'erreur (ici en commun avec la sortie)
#SBATCH --constraint=a100  #partition 4GPU V100-32Go
#SBATCH --ntasks=1                          # Utilisation de 2 n ^suds
#SBATCH --ntasks-per-node=1                         # 1 t  che par n ^sud
#SBATCH --gres=gpu:1                       # 1 GPU par n ^sud, donc 2 GPU au total
#SBATCH --cpus-per-task=8                 # On r  serve 10 cores CPU par t  che (ajuster selon les besoins de votre application)
#SBATCH --time=03:00:00
#SBATCH --hint=nomultithread
#SBATCH --account=zsl@a100

set -e  # Stop script when an error occurs

module purge

module load cpuarch/amd
module load pytorch-gpu/py3/2.1.1

# echo des commandes lancees
set -x


MODEL_PATH="models/bert-base-uncased"
MODEL_NAME="bert-base-uncased"

srun -l python3 train.py --data-path ./data \
                --model-path ${MODEL_PATH} \
                --model-name ${MODEL_NAME} \
                --output-dir ./output \
                --batch-size 32 \
                --num-epochs 5 \
                --device cuda \
                --seed 1 \
                --jeanzay \
                --offline

