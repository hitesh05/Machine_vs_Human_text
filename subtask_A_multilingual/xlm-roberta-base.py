import json
from tqdm import tqdm
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, get_linear_schedule_with_warmup
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import numpy as np
from sklearn.metrics import f1_score, accuracy_score

# Load data
print('Loading dataframe')
train_data = pd.read_json('autext/data/SemEval8/SubtaskA/subtaskA_train_multilingual.jsonl', lines=True)
dev_data = pd.read_json('autext/data/SemEval8/SubtaskA/subtaskA_dev_multilingual.jsonl', lines=True)

train_data = train_data.sample(n=500)
dev_data = dev_data.sample(n=100)

# Define model and training parameters
model_name = "xlm-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

max_sequence_length = 512
batch_size = 8
learning_rate = 2e-5
num_epochs = 3
warmup_proportion = 0.1
num_labels = 2

# Tokenize and preprocess data
print('Tokenizing data')
def tokenize_and_pad_text(text, max_length):
    tokens = tokenizer.tokenize(text)
    if len(tokens) > max_length - 2:  # for [CLS] and [SEP] tokens
        tokens = tokens[:max_length - 2]

    input_ids = tokenizer.convert_tokens_to_ids(['[CLS]'] + tokens + ['[SEP]'])
    input_ids += [0] * (max_length - len(input_ids))  # Pad with zeros
    return input_ids

train_data['tokenized_text'] = train_data['text'].apply(lambda x: tokenize_and_pad_text(x, max_sequence_length))
dev_data['tokenized_text'] = dev_data['text'].apply(lambda x: tokenize_and_pad_text(x, max_sequence_length))

train_inputs = torch.tensor(train_data['tokenized_text'].values.tolist())
dev_inputs = torch.tensor(dev_data['tokenized_text'].values.tolist())

train_labels = torch.tensor(train_data['label'].values)
dev_labels = torch.tensor(dev_data['label'].values)

train_dataset = TensorDataset(train_inputs, train_labels)
dev_dataset = TensorDataset(dev_inputs, dev_labels)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size)

# Initialize and configure the model
print('Initializing model')
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
model.to(device)

# Set up the optimizer and learning rate scheduler
optimizer = AdamW(model.parameters(), lr=learning_rate)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(len(train_dataloader) * warmup_proportion), num_training_steps=len(train_dataloader) * num_epochs)

criterion = nn.CrossEntropyLoss()

# Training loop
print('Training model')
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", ncols=100)

    for batch in progress_bar:
        batch_inputs, batch_labels = batch
        batch_inputs = batch_inputs.to(device)
        batch_labels = batch_labels.to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=batch_inputs, labels=batch_labels)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        progress_bar.set_postfix(loss=loss.item())

    progress_bar.close()
    avg_train_loss = total_loss / len(train_dataloader)

    # Evaluation on the validation set
    model.eval()
    with torch.no_grad():
        true_labels = []
        predicted_labels = []

        for batch in dev_dataloader:
            batch_inputs, batch_labels = batch
            batch_inputs = batch_inputs.to(device)
            batch_labels = batch_labels.to(device)

            outputs = model(input_ids=batch_inputs)
            logits = outputs.logits

            _, predicted = torch.max(logits, dim=1)
            true_labels.extend(batch_labels.cpu().numpy())
            predicted_labels.extend(predicted.cpu().numpy())

        accuracy = accuracy_score(true_labels, predicted_labels)
        f1 = f1_score(true_labels, predicted_labels)

    print(f"Epoch {epoch + 1}/{num_epochs} - Avg. Train Loss: {avg_train_loss:.4f} - Validation Accuracy: {accuracy:.4f} - F1 Score: {f1:.4f}")

    with open("output_subtaskA_multi_xlmrobertabase.txt", "a") as output_file:
        output_line = f"Epoch {epoch + 1}/{num_epochs} - Avg. Train Loss: {avg_train_loss:.4f} - Validation Accuracy: {accuracy:.4f} - F1 Score: {f1:.4f}\n"
        output_file.write(output_line)

# Save the trained model
torch.save(model.state_dict(), 'subtaskA_multi_xlmrobertabase.pth')