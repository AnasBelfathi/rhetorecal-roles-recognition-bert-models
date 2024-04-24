# train_model.sh
#!/bin/bash

MODEL_PATH="models/bert-base-uncased"
MODEL_NAME="bert-base-uncased"

# Loop over seeds from 1 to 5 and run the training script
for SEED in {1..5}; do
    echo "Running training with seed $SEED"
    python3 train.py --data-path ./data \
                    --model-path ${MODEL_PATH} \
                    --model-name ${MODEL_NAME} \
                    --output-dir ./output \
                    --batch-size 32 \
                    --num-epochs 5 \
                    --device cuda \
                    --seed $SEED \
                    --offline \
                    --use-mini-dataset
done
