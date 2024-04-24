import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Run the model training and evaluation")

    parser.add_argument("--data-path", type=str, default="./data", help="Path to the data directory")
    parser.add_argument("--model-path", type=str, default="bert-base-uncased", help="Model path")
    parser.add_argument("--model-name", type=str, default="bert-base-uncased", help="Model name")
    parser.add_argument("--output-dir", type=str, default="./output", help="Directory for output files")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for training and evaluation")
    parser.add_argument("--learning-rate", type=int, default=3e-5, help="Learning Rate for training")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1, help="Number of updates steps to accumulate before")
    parser.add_argument("--num-epochs", type=int, default=3, help="Number of epochs for training")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for training ('cuda', 'cpu', 'mps')")
    parser.add_argument("--use-mini-dataset", action="store_true", help="Use a mini dataset for quick testing")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle training data")
    parser.add_argument("--jeanzay", action="store_true", help="Using the jeanzay server")
    parser.add_argument("--offline", action="store_true", help="Load the model and data using the offline mode")


    # # Ajout d'un argument pour sélectionner un sous-ensemble pour un mini-test
    # parser.add_argument("--seed", type=int, default=42,
    #                     help="Seed for initializing random number generators for reproducibility")



    return parser.parse_args()


# config.py
rr_labels = [
    'PREAMBLE', 'FAC', 'RLC', 'ISSUE', 'ARG_PETITIONER', 'ARG_RESPONDENT',
    'ANALYSIS', 'STA', 'PRE_RELIED', 'PRE_NOT_RELIED', 'RATIO', 'RPC', 'NONE'
]

label2id = {label: i for i, label in enumerate(rr_labels)}
id2label = {i: label for label, i in label2id.items()}
