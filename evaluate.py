from datasets import load_dataset
from transformers import pipeline
from config import parse_args


def evaluate_and_predict(model_path, data_path):
    classifier = pipeline("text-classification", model=model_path, tokenizer=model_path)
    test_dataset = load_dataset(data_path, data_files={'test': 'test.csv'})['test']

    # Exemple de prédiction sur le premier échantillon
    prediction = classifier(test_dataset[0]['text'])
    print(prediction)


if __name__ == "__main__":
    args = parse_args()
    evaluate_and_predict(args.output_dir, args.data_path)
