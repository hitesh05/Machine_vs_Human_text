# -*- coding: utf-8 -*-
"""Copy of longformer+pytorchlightning.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/11EefNhlprSa22PCCYyxT_XU8cpWsc8YO
"""

# from google.colab import drive
# drive.mount('/content/drive')

# !pip install transformers[torch]
# !pip install lightning

# !pip install transformers
# !pip install datasets
# !pip install wandb

import torch
import torchtext
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, BertModel
from transformers import AutoTokenizer, LongformerModel, LongformerTokenizer
import pytorch_lightning as pl
from transformers import LongformerTokenizerFast, LongformerForSequenceClassification, Trainer, TrainingArguments, LongformerConfig

# def tokenize_and_pad_text(text, max_length,tokenizer):
#     tokens = tokenizer.tokenize(text)
#     if len(tokens) > max_length - 2:  # for [CLS] and [SEP] tokens
#         tokens = tokens[:max_length - 2]

#     # Add special tokens ([CLS] and [SEP]) and pad to max_length
#     input_ids = tokenizer.convert_tokens_to_ids(['[CLS]'] + tokens + ['[SEP]'])
#     input_ids += [0] * (max_length - len(input_ids))  # Pad with zeros
#     return input_ids

def tokenize_and_pad_text(text, max_length):
    encoded = tokenizer.encode_plus(
        text,
        max_length=max_length,
        add_special_tokens=True,
        truncation=True,
        padding='max_length',
        return_tensors='pt'
    )
    input_ids = encoded['input_ids']
    return input_ids

class LongformerClassifier(pl.LightningModule):
    def __init__(self, num_classes, tokenizer, model_name="allenai/longformer-base-4096"):
        super().__init__()
        self.longformer = LongformerModel.from_pretrained(model_name)
        # Freeze the bottom 8 layers
        for param in self.longformer.base_model.parameters()[:8]:
            param.requires_grad = False
        self.bi_lstm = nn.LSTM(self.longformer.config.hidden_size,
                               self.longformer.config.hidden_size // 2,
                               batch_first=True,
                               bidirectional=True)
        self.classifier = torch.nn.Linear(self.longformer.config.hidden_size, num_classes)
        self.tokenizer = tokenizer

    def forward(self, input_ids, attention_mask):
        outputs = self.longformer(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state

        lstm_out, _ = self.bi_lstm(last_hidden_state)
        logits = self.classifier(lstm_out[:, -1, :])  # Use the last hidden state for classification

        return logits

    def training_step(self, batch, batch_idx):
        input_ids, labels = batch
        attention_mask = (input_ids != self.tokenizer.pad_token_id).float()
        logits = self(input_ids=input_ids, attention_mask=attention_mask)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, labels = batch
        attention_mask = (input_ids != self.tokenizer.pad_token_id).float()
        logits = self(input_ids=input_ids, attention_mask=attention_mask)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        self.log('val_loss', loss)
        preds = torch.argmax(logits, dim=1)
        acc = torch.sum(preds == labels) / len(preds)
        self.log('val_acc', acc)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-5)
if __name__ == '__main__':
        
    mono_train_data = pd.read_json('data/SemEval8/SubtaskA/subtaskA_train_monolingual.jsonl', lines=True)
    mono_dev_data = pd.read_json('data/SemEval8/SubtaskA/subtaskA_dev_monolingual.jsonl', lines=True)
    #tokenizer = LongformerTokenizerFast.from_pretrained('allenai/longformer-base-4096', max_length = 1024)
    model_name = "allenai/longformer-base-4096"
    tokenizer = LongformerTokenizer.from_pretrained(model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    max_sequence_length = 4096
    
    mono_train_data = mono_train_data

    mono_dev_data = mono_dev_data[900:1100]

    mono_train_data['tokenized_text'] = mono_train_data['text'].apply(lambda x: tokenize_and_pad_text(x, max_sequence_length,tokenizer))
    mono_dev_data['tokenized_text'] = mono_dev_data['text'].apply(lambda x: tokenize_and_pad_text(x, max_sequence_length,tokenizer))

    train_inputs = torch.stack([torch.tensor(arr).detach() for arr in mono_train_data['tokenized_text'].values])
    dev_inputs = torch.stack([torch.tensor(arr).detach() for arr in mono_dev_data['tokenized_text'].values])

    train_labels = torch.tensor(mono_train_data['label'].values)
    dev_labels = torch.tensor(mono_dev_data['label'].values)

    # Create TensorDatasets
    train_dataset = TensorDataset(train_inputs, train_labels)
    dev_dataset = TensorDataset(dev_inputs, dev_labels)

    batch_size = 1

    # Create DataLoaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)

    num_classes = 2
    model = LongformerClassifier(num_classes=num_classes, model_name="allenai/longformer-base-4096",tokenizer=tokenizer)
    # trainer = pl.Trainer(max_epochs=3, gpus=1)
    trainer = pl.Trainer(max_epochs=2, accelerator="auto")

    trainer.fit(model, train_dataloader, dev_dataloader)
    trainer.save_checkpoint("main_model.ckpt")
