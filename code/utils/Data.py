import os
import pickle
from glob import glob
import pandas as pd
from transformers import AutoTokenizer
from datasets import Dataset, ClassLabel
from config import config
from tqdm import tqdm

def tokenize_function(examples, llm, parent_text=True):
    tokenizer = AutoTokenizer.from_pretrained(llm)

    if parent_text:
        tokenized_inputs = tokenizer(text=examples["text"], text_pair=examples["parent_text"], padding="max_length", truncation=True, max_length=200)
    else:
        tokenized_inputs = tokenizer(text=examples["text"], padding="max_length", truncation=True, max_length=200)
    
    return tokenized_inputs
    
def generate_data(profile, split, data_dir, llm, classes, parent_text):
    filepath = os.path.join(data_dir, f'{profile}_{split}_set.csv')
    print('----->>>> ',filepath)
    data = pd.read_csv(filepath, dtype=object)
    labels = [1 if i == classes[1] else 0 for i in data['label'].tolist()]
    data['labels'] = labels
    dt = Dataset.from_pandas(data)
    if parent_text:
        tokenized_dataset = dt.map(lambda x:tokenize_function(x, llm, parent_text), remove_columns=['text', 'label', 'parent_text'])
    else:
        tokenized_dataset = dt.map(lambda x:tokenize_function(x, llm, parent_text), remove_columns=['text', 'label'])
    tokenized_dataset = tokenized_dataset.with_format("torch")
    return  tokenized_dataset
    
def tokenize_inference(text, llm, parent_text=True):
    tokenizer = AutoTokenizer.from_pretrained(llm)

    tokenized = tokenizer.encode_plus(text, return_tensors='pt')
    
    return tokenized
