from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report, f1_score
import pandas as pd
import logging
import os
import json

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted', zero_division=0)
    return {"precision": precision, "recall": recall, "f1": f1}

def set_seed(seed=42):
    import numpy as np
    import torch
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def calculate_metrics_and_generate_report(predictions_path, label2id, output_dir):
    # Charger les prédictions depuis le fichier CSV
    predictions_df = pd.read_csv(predictions_path)

    # Assurez-vous que 'True Label' est dans le format attendu (par exemple, des identifiants numériques si nécessaire)
    # Et que 'Predictions' contient des labels textuels comme les clés dans label2id

    # Il est important de s'assurer que chaque label textuel dans 'Predictions' a un correspondant dans label2id
    predictions_df['Pred Label ID'] = predictions_df['Predictions'].map(label2id)
    predictions_df['True Label ID'] = predictions_df['True Label'].map(label2id)

    # Générer une liste de tous les labels (identifiants numériques) attendus
    labels = list(label2id.values())

    report = classification_report(predictions_df['True Label ID'], predictions_df['Pred Label ID'],
                                   labels=labels,  # Fournir explicitement les identifiants de labels attendus
                                   target_names=list(label2id.keys()),  # Noms des labels pour l'affichage
                                   zero_division=0)
    print(report)
    micro_f1 = f1_score(predictions_df['True Label ID'], predictions_df['Pred Label ID'], average='micro')
    macro_f1 = f1_score(predictions_df['True Label ID'], predictions_df['Pred Label ID'], average='macro')

    metrics = {
        "micro_f1": micro_f1,
        "macro_f1": macro_f1
    }

    metrics_file_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_file_path, "w") as json_file:
        json.dump(metrics, json_file, indent=4)

    return metrics




def log_best_model(trainer):
    logging.info(f"Best model directory: {trainer.state.best_model_checkpoint}")
    logging.info(f"Best model score: {trainer.state.best_metric}")