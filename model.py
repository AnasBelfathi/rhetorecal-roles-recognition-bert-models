from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import pipeline
from datasets import load_metric
import os
import csv
from utils import calculate_metrics_and_generate_report
from config import id2label
def get_model_and_tokenizer(model_name, label2id):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # print(len(label2id))
    # print({label: str(i) for i, label in enumerate(label2id)})
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label2id),
        id2label={str(i): label for i, label in enumerate(label2id)},
        label2id={label: str(i) for i, label in enumerate(label2id)},
    )
    return model, tokenizer


def evaluate_and_save_predictions(model, tokenizer, test_dataset, output_dir, label2id):
    classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, return_all_scores=False)

    predictions_file_path = os.path.join(output_dir, "predictions.csv")
    with open(predictions_file_path, mode='w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Text', 'Predictions', 'True Label'])

        for example in test_dataset:
            # Prédiction pour l'exemple courant
            pred = classifier(example["text"])
            top_prediction = pred[0]['label']

            # Écrivez les résultats dans le fichier CSV
            csv_writer.writerow([example['text'], top_prediction, id2label[example['labels']]])

    # Calcul et affichage des métriques et du rapport de classification
    return calculate_metrics_and_generate_report(predictions_file_path, label2id, output_dir)
