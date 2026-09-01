import pandas as pd
from pathlib import Path
from .BERTTicketsDataset import BERTTicketsDataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import json

MODEL_NAME = "google-bert/bert-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

project_root = Path(__file__).parent.parent.parent
data_root = project_root / 'data'
output_root = project_root / 'outputs'
train_path = data_root / 'train.csv'
valid_path = data_root / 'valid.csv'
test_path = data_root / 'test.csv'
max_len_path = output_root / 'decide_max_len' / 'max_len.json'

train_file = pd.read_csv(train_path)
valid_file = pd.read_csv(valid_path)
test_file = pd.read_csv(test_path)
with max_len_path.open('r',encoding='utf-8') as max_len_file:
    max_len = json.load(max_len_file)['max_len']

labels = sorted(train_file["label"].unique())
label_to_id = {
    label: index
    for index, label in enumerate(labels)
}

train_dataset = BERTTicketsDataset(texts=train_file['text'].tolist(),
                                    labels=train_file['label'].map(label_to_id).tolist(),
                                    tokenizer=tokenizer,
                                    max_len=max_len)

valid_dataset = BERTTicketsDataset(texts=valid_file['text'].tolist(),
                                    labels=valid_file['label'].map(label_to_id).tolist(),
                                    tokenizer=tokenizer,
                                    max_len=max_len)

test_dataset = BERTTicketsDataset(texts=test_file['text'].tolist(),
                                    labels=test_file['label'].map(label_to_id).tolist(),
                                    tokenizer=tokenizer,
                                    max_len=max_len)
'''test_file['label'].map(label_to_id)：把每个文字label转化成数字，与编码后的text格式相同'''

batch_size = 8
train_dataloader = DataLoader(train_dataset,batch_size,shuffle=True)
valid_dataloader = DataLoader(valid_dataset,batch_size,shuffle=False)
test_dataloader = DataLoader(test_dataset,batch_size,shuffle=False)


def prepare_data(if_train):
    if if_train:
        return {
            'train_dataloader':train_dataloader,
            'valid_dataloader':valid_dataloader,
            'label_to_id':label_to_id
        }
    else:
        return{
            'test_dataloader':test_dataloader,
            'label_to_id':label_to_id
        }

    