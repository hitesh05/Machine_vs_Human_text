# subtaskA
# monolingual
# bert-base-uncased
# batchsize = 8
# learning_rate = 2e-5
# num_epochs = 5 
# warmup_proportion = 0.1
# adamW optimizer
# crossEntropyLoss

import json
from tqdm import tqdm
import pandas as pd
from transformers import AutoTokenizer

mono_train_data = pd.read_json('SemEval8/SubtaskA/subtaskA_train_monolingual.jsonl', lines=True)
mono_dev_data = pd.read_json('SemEval8/SubtaskA/subtaskA_dev_monolingual.jsonl', lines=True)

'''
models to tryout
1. bert-base-uncased
2. bert-base-cased
3. roberta-base
4. gpt2
5. microsoft/deberta-base
'''

model_name = "bert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_name)

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def tokenize_and_pad_text(text, max_length):
    tokens = tokenizer.tokenize(text)
    if len(tokens) > max_length - 2:  # for [CLS] and [SEP] tokens
        tokens = tokens[:max_length - 2]
    
    # Add special tokens ([CLS] and [SEP]) and pad to max_length
    input_ids = tokenizer.convert_tokens_to_ids(['[CLS]'] + tokens + ['[SEP]'])
    input_ids += [0] * (max_length - len(input_ids))  # Pad with zeros
    return input_ids

max_sequence_length = 512

mono_train_data['tokenized_text'] = mono_train_data['text'].apply(lambda x: tokenize_and_pad_text(x, max_sequence_length))
mono_dev_data['tokenized_text'] = mono_dev_data['text'].apply(lambda x: tokenize_and_pad_text(x, max_sequence_length))

train_inputs = torch.tensor(mono_train_data['tokenized_text'].values.tolist())
dev_inputs = torch.tensor(mono_dev_data['tokenized_text'].values.tolist())

from torch.utils.data import DataLoader, TensorDataset

train_labels = mono_train_data['label'].values
dev_labels = mono_dev_data['label'].values

train_dataset = TensorDataset(train_inputs, torch.tensor(train_labels))
dev_dataset = TensorDataset(dev_inputs, torch.tensor(dev_labels))

batch_size = 8

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW


num_labels = 2

# tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
model.to(device)

learning_rate = 2e-5
num_epochs = 5 
warmup_proportion = 0.1

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

num_training_steps = len(train_dataloader) * num_epochs
num_warmup_steps = int(num_training_steps * warmup_proportion)

scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=num_warmup_steps, T_mult=1)

criterion = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", ncols=100)

    for batch in progress_bar:
    # for batch in train_dataloader:
        batch_inputs, batch_labels = batch
        batch_inputs = batch_inputs.to(device)
        batch_labels = batch_labels.to(device)

        optimizer.zero_grad()
        
        # attention_mask = (batch_inputs != 0).float()

        # outputs = model(input_ids=batch_inputs, attention_mask=attention_mask, labels=batch_labels)
        outputs = model(input_ids=batch_inputs, labels=batch_labels)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        
        progress_bar.set_postfix(loss=loss.item())

    progress_bar.close()
    # Calculate average training loss for this epoch
    avg_train_loss = total_loss / len(train_dataloader)

    # Evaluation on the validation set
    model.eval()
    with torch.no_grad():
        correct_predictions = 0
        total_predictions = 0

        for batch in dev_dataloader:
            batch_inputs, batch_labels = batch
            batch_inputs = batch_inputs.to(device)
            batch_labels = batch_labels.to(device)

            # attention_mask = (batch_inputs != 0).float()

            # outputs = model(input_ids=batch_inputs, attention_mask=attention_mask)
            outputs = model(input_ids=batch_inputs)
            logits = outputs.logits

            _, predicted = torch.max(logits, dim=1)
            total_predictions += batch_labels.size(0)
            correct_predictions += (predicted == batch_labels).sum().item()

        accuracy = correct_predictions / total_predictions

    print(f"Epoch {epoch + 1}/{num_epochs} - Avg. Train Loss: {avg_train_loss:.4f} - Validation Accuracy: {accuracy:.4f}")
    with open("output_subtaskA_mono_bertbaseuncased.txt", "a") as output_file:
      output_line = f"Epoch {epoch + 1}/{num_epochs} - Avg. Train Loss: {avg_train_loss:.4f} - Validation Accuracy: {accuracy:.4f}\n"
      output_file.write(output_line)
      
torch.save(model.state_dict(), 'subtaskA_mono_bertbaseuncased.pth')
