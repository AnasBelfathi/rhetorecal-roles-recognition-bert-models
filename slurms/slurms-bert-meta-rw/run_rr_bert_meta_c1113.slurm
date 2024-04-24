#!/bin/bash
#SBATCH --job-name=rr_bert_meta_c1113
#SBATCH --output=./job_out_err/rr_bert_meta_c1113_%A_%a.out
#SBATCH --error=./job_out_err/rr_bert_meta_c1113_%A_%a.err
#SBATCH --constraint=v100-32g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time=10:00:00
#SBATCH --hint=nomultithread
#SBATCH --account=bvh@v100
#SBATCH --array=1-5%5
#SBATCH --mail-user=anas.belfathi@etu.univ-nantes.fr
#SBATCH --mail-type=ALL

module purge
module load pytorch-gpu/py3/2.1.1

ROOT="/gpfswork/rech/bvh/commun/legal-masking"

MODEL_NAME="bert-base-uncased-jzNEW-16-4-4-e20-b32-c512-metadiscourse-weighted_random-exall"
CHECKPOINT="1113"
OUTPUT_MODEL_NAME="bert-meta-rw-c$CHECKPOINT"


MODEL_PATH="$ROOT/models/$MODEL_NAME/checkpoint-$CHECKPOINT"


DATA_PATH="./data"

SEED=$SLURM_ARRAY_TASK_ID  # Utilisez l'ID de tâche du tableau pour le seed
OUTPUT_DIR="./output"

srun python3 train.py --data-path ${DATA_PATH} \
                --model-path ${MODEL_PATH} \
                --model-name ${OUTPUT_MODEL_NAME} \
                --output-dir ${OUTPUT_DIR} \
                --batch-size 32 \
                --num-epochs 5 \
                --device cuda \
                --seed ${SEED} \
                --offline \
                --jeanzay