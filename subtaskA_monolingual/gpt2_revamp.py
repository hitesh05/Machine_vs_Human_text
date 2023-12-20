import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# from google.colab import drive
# drive.mount('/content/drive')

import os
os.environ["WANDB_DISABLED"] = "true"
seed = 25
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

train_data = pd.read_json('SemEval8/SubtaskA/subtaskA_train_monolingual.jsonl', lines=True)
train_data = train_data.reset_index(drop=True)
dev_data = pd.read_json('SemEval8/SubtaskA/subtaskA_dev_monolingual.jsonl', lines=True)
dev_data = dev_data.reset_index(drop=True)

from sklearn.model_selection import train_test_split
train_texts = train_data['text'].to_list()[:12000]
train_labels = train_data['label'].to_list()[:12000]

val_texts = train_data['text'].to_list()[:2000]
val_labels = train_data['label'].to_list()[:2000]

print(len(train_texts))
print(len(val_texts))

# !pip install transformers

from transformers import AdamW, AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from transformers import TrainingArguments, Trainer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForSequenceClassification.from_pretrained("gpt2").to(device)
model.config.pad_token_id = model.config.eos_token_id
print("Checking Model Configurations")
print()
print(model.config)

train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=256)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=256)

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels[idx] == 0:
            item['labels'] = torch.tensor(0)
        else:
            item['labels'] = torch.tensor(1)
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = CustomDataset(train_encodings, train_labels)
val_dataset = CustomDataset(val_encodings, val_labels)

print("Sample Data Point")
print()
print(train_dataset[0])

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='macro')
    acc = accuracy_score(labels, predictions)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# !pip install accelerate -U
# !pip install transformers[torch]

training_args = TrainingArguments(
    output_dir='./SubtaskA_gpt_models/',
    num_train_epochs=5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model='accuracy',
    evaluation_strategy='epoch',
    save_strategy='epoch',
    save_total_limit=1,
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)
trainer.train()

# trainer.evaluate(test_dataset)
# predictions, labels, _ = trainer.predict(test_dataset)
# predictions = np.argmax(predictions, axis=1)

# from sklearn.metrics import classification_report
# print(classification_report(labels, predictions))

trainer.save_model('./BOOM_them_GPT_models/english_gpt2_task1/trained_model')

