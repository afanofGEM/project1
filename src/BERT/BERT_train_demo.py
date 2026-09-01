from pathlib import Path
import torch
import torch.nn as nn
from .BERTTicketsDataLoader import prepare_data
from transformers import AutoModelForSequenceClassification

MODEL_NAME = "google-bert/bert-base-chinese"
if_train = True

'''{
    'train_dataloader':train_dataloader,
    'valid_dataloader':valid_dataloader
}'''
dataloader_dict = prepare_data(if_train)
train_dataloader = dataloader_dict['train_dataloader']
valid_dataloader = dataloader_dict['valid_dataloader']

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=5,
)

for batch in train_dataloader:
    model.train()
    '''{
            'input_id':encoded['input_ids'].squeeze(0), #tensor(max_len)
            'label_id':torch.tensor(label,dtype=torch.long), #tensor(1)
            'padding_mask':encoded['attention_mask'].squeeze(0)
        }
    合体之后input_ids:tensor(batch_size,max_len)
    label_id:tensor(batch_size)
    padding_mask:tensor(batch_size,max_len)'''

    input_ids = batch["input_ids"]
    attention_mask = batch["padding_mask"]
    label_ids = batch["label_ids"]

    outputs = model(input_ids=input_ids,attention_mask=attention_mask,labels=label_ids)
    logits = outputs.logits #(batch_size,num_labels)
    loss = outputs.loss

    print("input_ids.shape:", input_ids.shape)
    print("attention_mask.shape:", attention_mask.shape)
    print("labels.shape:", label_ids.shape)
    print("logits.shape:", logits.shape)
    print("loss:", loss.item())