import pathlib
import random
import sys
import pandas as pd
import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import TensorDataset, DataLoader
from transformers import RobertaTokenizer

from features.probabilistic import ProbabilisticFeatures, fixed_len
from models.hybrid import HybridBiLSTMRoBERTa
from models.training import eval_loop, train_loop

random.seed(10)
torch.manual_seed(10)
np.random.seed(0)

language = 'en' # english/multilingual
task = 'subtask_1' # 1/2
if len(sys.argv) == 3:
    language = sys.argv[1]
    task = sys.argv[2]

model_type = 'Hybrid'
shuffle_traindev = False

if language == 'en':
    roberta_variant = "roberta-base"
else:
    roberta_variant = 'xlm-roberta-base'

train_data = None
dev_data = None

if task == 'subtask_1' and language == 'en':
    train_data = pd.read_json('data/SemEval8/SubtaskA/subtaskA_train_monolingual.jsonl', lines=True)
    dev_data = pd.read_json('data/SemEval8/SubtaskA/subtaskA_dev_monolingual.jsonl', lines=True)
elif task == 'subtask_1' and language != 'en':
    train_data = pd.read_json('data/SemEval8/SubtaskA/subtaskA_train_multilingual.jsonl', lines=True)
    dev_data = pd.read_json('data/SemEval8/SubtaskA/subtaskA_dev_multilingual.jsonl', lines=True)
elif task == 'subtask_2':
    train_data = pd.read_json('data/SemEval8/SubtaskB/subtaskB_train.jsonl', lines=True)
    dev_data = pd.read_json('data/SemEval8/SubtaskB/subtaskB_dev.jsonl', lines=True)

train_text = []
train_Y = []
# train_ids = []
dev_text = []
dev_Y = []
# dev_ids = []

partial_sents = 1000

if task == 'subtask_1':
    for ind, row in train_data.iterrows():
        train_text.append(row['text'].strip())
        train_Y.append(int(row['label']))
        # if ind > partial_sents:
        #     break
    for ind, row in dev_data.iterrows():
        dev_data.append(row['text'].strip())
        train_Y.append(int(row['label']))
        # if ind > partial_sents/10:
        #     break

train_Y = np.array(train_Y)
dev_Y = np.array(dev_Y)

print("Loaded data with " + str(len(train_Y) + len(dev_Y)) + " instances.")

# Preparing feature generators
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
local_device = torch.device('cpu')

perp = ProbabilisticFeatures(device, local_device, language)
feature_generators = [perp]

print("Generating sequence features...")
train_X = []
dev_X = []
# test_X = []
for in_text, out_X in zip([train_text, dev_text], [train_X, dev_X]):
    for feature_generator in feature_generators:
        out_X.append(np.array(feature_generator.word_features(in_text)))
train_X = np.concatenate(train_X, axis=2)
dev_X = np.concatenate(dev_X, axis=2)
# test_X = np.concatenate(test_X, axis=2)

print("Tokenising text for RoBERTa...")
tokenizer = RobertaTokenizer.from_pretrained(roberta_variant)
train_encodings = tokenizer(train_text, padding=True, truncation=True, max_length=fixed_len, return_tensors="pt")
dev_encodings = tokenizer(dev_text, padding=True, truncation=True, max_length=fixed_len, return_tensors="pt")
# test_encodings = tokenizer(test_text, padding=True, truncation=True, max_length=fixed_len, return_tensors="pt")

# CUDA memory cleaning
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print(str(torch.cuda.get_device_properties(0).total_memory))
    print(str(torch.cuda.memory_reserved(0)))
    print(str(torch.cuda.memory_allocated(0)))

train_input_ids = train_encodings['input_ids']
train_attention_mask = train_encodings['attention_mask']

dev_input_ids = dev_encodings['input_ids']
dev_attention_mask = dev_encodings['attention_mask']

# test_input_ids = test_encodings['input_ids']
# test_attention_mask = test_encodings['attention_mask']

print("Building a model...")
BATCH_SIZE = 4
train_dataset = TensorDataset(torch.tensor(train_X).float(), train_input_ids, train_attention_mask,
                              torch.tensor(np.array(train_Y)).long())
dev_dataset = TensorDataset(torch.tensor(dev_X).float(), dev_input_ids, dev_attention_mask,
                            torch.tensor(np.array(dev_Y)).long())
# test_dataset = TensorDataset(torch.tensor(test_X).float(), test_input_ids, test_attention_mask,
#                              torch.tensor(np.zeros(len(test_text))).long())
train_loader = DataLoader(train_dataset, shuffle=shuffle_traindev, batch_size=BATCH_SIZE)
dev_loader = DataLoader(dev_dataset, shuffle=shuffle_traindev, batch_size=BATCH_SIZE)
# test_loader = DataLoader(test_dataset, shuffle=False, batch_size=BATCH_SIZE)

model = HybridBiLSTMRoBERTa(train_X.shape[2], task, local_device, roberta_variant).to(device)

print("Preparing training")
model = model.to(device)
learning_rate = 1e-3
optimizer = Adam(model.parameters(), lr=learning_rate)
milestones = [5] if model_type == 'Hybrid' else []
scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.02)
skip_visual = False
stats_file = open('main_autex_1.txt', 'w')
stats_file.write('epoch\ttrain_F1\tdev_F1\n')

eval_loop(dev_loader, model, device, local_device, skip_visual)
for epoch in range(20):
    print("EPOCH " + str(epoch + 1))
    if model_type == 'Hybrid':
        if epoch < 5:
            model.freeze_llm()
        else:
            model.unfreeze_llm()
    train_f1 = train_loop(train_loader, model, optimizer, scheduler, device, local_device, skip_visual)
    dev_f1 = eval_loop(dev_loader, model, device, local_device, skip_visual, test=False)
    # test_preds, test_probs = eval_loop(test_loader, model, device, local_device, skip_visual, test=True)
    
    stats_file.write(str(epoch + 1) + '\t' + str(train_f1) + '\t' + str(dev_f1) + '\n')
stats_file.close()
print("The end!")
