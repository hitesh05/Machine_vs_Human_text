import torch
import torchtext
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, BertModel
from transformers import AutoTokenizer, LongformerModel, LongformerTokenizer
import pytorch_lightning as pl

mono_train_data = pd.read_json('SemEval8/SubtaskA/subtaskA_train_monolingual.jsonl', lines=True)
mono_dev_data = pd.read_json('SemEval8/SubtaskA/subtaskA_dev_monolingual.jsonl', lines=True)

model_name = "allenai/longformer-base-4096"
tokenizer = LongformerTokenizer.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def tokenize_and_pad_text(text, max_length):
    tokens = tokenizer.tokenize(text)
    if len(tokens) > max_length - 2:  # for [CLS] and [SEP] tokens
        tokens = tokens[:max_length - 2]

    # Add special tokens ([CLS] and [SEP]) and pad to max_length
    input_ids = tokenizer.convert_tokens_to_ids(['[CLS]'] + tokens + ['[SEP]'])
    input_ids += [0] * (max_length - len(input_ids))  # Pad with zeros
    return input_ids

max_sequence_length = 4096

mono_train_data = mono_train_data[:7000]
mono_dev_data = mono_dev_data[:1000]

mono_train_data['tokenized_text'] = mono_train_data['text'].apply(lambda x: tokenize_and_pad_text(x, max_sequence_length))
mono_dev_data['tokenized_text'] = mono_dev_data['text'].apply(lambda x: tokenize_and_pad_text(x, max_sequence_length))

train_inputs = torch.tensor(mono_train_data['tokenized_text'].values.tolist())
dev_inputs = torch.tensor(mono_dev_data['tokenized_text'].values.tolist())

train_labels = torch.tensor(mono_train_data['label'].values)
dev_labels = torch.tensor(mono_dev_data['label'].values)

# Create TensorDatasets
train_dataset = TensorDataset(train_inputs, train_labels)
dev_dataset = TensorDataset(dev_inputs, dev_labels)

batch_size = 1

# Create DataLoaders
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)

class LongformerClassifier(pl.LightningModule):
    def __init__(self, num_classes, model_name="allenai/longformer-base-4096"):
        super().__init__()
        self.longformer = LongformerModel.from_pretrained(model_name)
        self.classifier = torch.nn.Linear(self.longformer.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.longformer(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        logits = self.classifier(last_hidden_state[:, 0, :])  # Use CLS token for classification
        return logits

    def training_step(self, batch, batch_idx):
        input_ids, labels = batch
        attention_mask = (input_ids != tokenizer.pad_token_id).float()
        logits = self(input_ids=input_ids, attention_mask=attention_mask)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, labels = batch
        attention_mask = (input_ids != tokenizer.pad_token_id).float()
        logits = self(input_ids=input_ids, attention_mask=attention_mask)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        self.log('val_loss', loss)
        preds = torch.argmax(logits, dim=1)
        acc = torch.sum(preds == labels) / len(preds)
        self.log('val_acc', acc)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-5)

num_classes = 2
model = LongformerClassifier(num_classes=num_classes, model_name="allenai/longformer-base-4096")
# trainer = pl.Trainer(max_epochs=3, gpus=1)
trainer = pl.Trainer(max_epochs=3, accelerator="auto")

trainer.fit(model, train_dataloader, dev_dataloader)

