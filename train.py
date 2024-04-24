import csv

import torch
from transformers import Trainer, TrainingArguments
from config import parse_args, label2id, id2label
from model import get_model_and_tokenizer, evaluate_and_save_predictions
from data import load_and_prepare_data
from utils import compute_metrics, set_seed, log_best_model
import logging
from datetime import datetime
import os
import torch

import torch.distributed as dist


# Configuration du logger
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
log_file = os.path.join(log_dir, f"training_{timestamp}.log")

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p',
                    handlers=[logging.FileHandler(log_file, mode='a'),
                              logging.StreamHandler()])

logger = logging.getLogger(__name__)





def main():
    args = parse_args()
    set_seed(args.seed)





    # model_name_short = args.model_name.split("/")[-1]

    output_dir = f"{args.output_dir}/{args.model_name}-bs{args.batch_size}-epochs{args.num_epochs}-minidata-{args.use_mini_dataset}-shuffle-{args.shuffle}/seed_{args.seed}"
    os.makedirs(output_dir, exist_ok=True)


    # Configuration des arguments d'entraînement
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to="tensorboard"
    )
    # For Jean Zay Server
    if args.jeanzay:
        import idr_torch
        dist.init_process_group(backend='nccl',
                                init_method='env://',
                                world_size=idr_torch.size,
                                rank=idr_torch.rank)
        training_args.local_rank = idr_torch.local_rank


    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )




    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # Chargement du modèle et du tokenizer
    model, tokenizer = get_model_and_tokenizer(args.model_path, label2id)

    # Chargement et préparation des données
    dataset = load_and_prepare_data(args.data_path, tokenizer, max_length=512, use_mini_dataset=args.use_mini_dataset, shuffle=args.shuffle, offline=args.offline)


    # Initialisation du Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        compute_metrics=compute_metrics,
    )

    # Entraînement
    trainer.train()
    log_best_model(trainer)
    tokenizer.save_pretrained(output_dir)

    # Assurez-vous de charger le meilleur modèle avant d'évaluer
    model_path = trainer.state.best_model_checkpoint
    model = model.from_pretrained(model_path)

    # Evaluation Process ...
    report = evaluate_and_save_predictions(model, tokenizer, dataset["test"], output_dir, label2id)

    # Results Printing
    print(f"Micro F1-score: {report['micro_f1']}")
    print(f"Macro F1-score: {report['macro_f1']}")


if __name__ == "__main__":
    main()
