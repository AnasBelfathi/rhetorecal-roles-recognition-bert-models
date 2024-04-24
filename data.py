import datasets
from datasets import load_dataset
from config import label2id
import logging
import pandas as pd
def load_and_prepare_data(data_path, tokenizer, max_length=512, use_mini_dataset=False, shuffle=False, offline=False):

    if offline:
        dataset = datasets.load_from_disk("./data-offline-mode")
    else:
        dataset = load_dataset(data_path, data_files={
           'train': 'train.csv',
           'test': 'test.csv',
           'validation': 'validation.csv'
        })
        dataset.save_to_disk("./data-offline-mode")


    # Si use_mini_dataset est True, réduire la taille du dataset
    if use_mini_dataset:
        for split in dataset.keys():
            dataset[split] = dataset[split].select(range(10))  # Exemple: 100 premiers échantillons

    if shuffle:
        for _ in range(3):
            dataset['train'] = dataset['train'].shuffle(seed=42) # attention: le seed peut afficher la même résultats pour l'entrainement

    print(dataset)
    # Ajout de cette fonction pour convertir les labels textuels en IDs
    def preprocess_labels(examples):
        print(label2id)
        examples['labels'] = [label2id[label] for label in examples['labels']]
        return examples

    dataset = dataset.map(preprocess_labels, batched=True)

    def tokenize_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=max_length)

    dataset = dataset.map(tokenize_function, batched=True)
    return dataset
