from pathlib import Path
import json

data_path = Path(__file__).parent.parent.parent / 'data'
dataframe_path = data_path / 'data.csv'
trainframe_path = data_path / 'train.csv'
validframe_path = data_path / 'valid.csv'
testframe_path = data_path / 'test.csv'
max_len_path = Path(__file__).parent.parent.parent / "outputs" /"decide_max_len" / "max_len.json"
mappings_path = Path(__file__).parent.parent.parent / "outputs" / "mappings"

with max_len_path.open('r',encoding='utf-8') as file:
    max_len = int(json.load(file)["max_len"])
batch_size = 8

#1. 读取数据
import pandas as pd
dataframe = pd.read_csv(dataframe_path)
trainframe = pd.read_csv(trainframe_path)
validframe = pd.read_csv(validframe_path)
testframe = pd.read_csv(testframe_path)

#2. 各数据集的dataset
def build_char_to_id(train_texts):
    char_to_id = {
        '<PAD>':0,
        '<UNK>':1,
    }

    for text in train_texts:
        for char in text:
            if char not in char_to_id:
                char_to_id[char] = len(char_to_id)

    return char_to_id


char_to_id = build_char_to_id(trainframe['text'].tolist())
labels = sorted(trainframe["label"].unique())

label_to_id = {
    label: index
    for index, label in enumerate(labels)
}


def save_mapping():
    """将训练时生成的映射表保存为可供推理服务读取的 JSON 文件。"""

    char_to_id_path = mappings_path / "char_to_id.json"

    with char_to_id_path.open('w',encoding='utf-8') as char_to_id_file:   
         json.dump(
            char_to_id,
            char_to_id_file,
            ensure_ascii=False, # 写中文的格式
            indent=2,# 缩进2格
         )

    label_to_id_path = mappings_path / "label_to_id.json"

    with label_to_id_path.open('w',encoding='utf-8') as label_to_id_file:   
         json.dump(
            label_to_id,
            label_to_id_file,
            ensure_ascii=False, # 写中文的格式
            indent=2,# 缩进2格
         )


def encode_text(text,char_to_id,max_len):

    '''先截断'''
    text = text[:max_len]
    result = []
    for char in text:
        if char in char_to_id:
            result.append(char_to_id[char])
        else:
            result.append(char_to_id['<UNK>'])

    '''padding的作用是：只有相同长度的单条数据才能被DataLoader合并'''
    while len(result) < max_len:
        result.append(char_to_id['<PAD>'])

    return result


def prepare_data(if_train):

    from common.dataset import TicketDataset
    traindataset = TicketDataset(texts=trainframe['text'].tolist(),
                                labels=trainframe['label'].tolist(),
                                char_to_idx=char_to_id,
                                label_to_idx=label_to_id,
                                max_len=max_len)

    validdataset = TicketDataset(texts=validframe['text'].tolist(),
                                labels=validframe['label'].tolist(),
                                char_to_idx=char_to_id,
                                label_to_idx=label_to_id,
                                max_len=max_len)

    testdataset = TicketDataset(texts=testframe['text'].tolist(),
                                labels=testframe['label'].tolist(),
                                char_to_idx=char_to_id,
                                label_to_idx=label_to_id,
                                max_len=max_len)

    '''你给dataset一个索引，它返回该条文本的
    {
        "text_idx": torch.tensor(text_idx, dtype=torch.long),
        "label_idx": torch.tensor(label_idx, dtype=torch.long)
    }'''

    #3.准备各数据集的dataloader
    from torch.utils.data import DataLoader
    train_dataloader = DataLoader(dataset=traindataset,
                                batch_size=batch_size,
                                shuffle=True)

    valid_dataloader = DataLoader(dataset=validdataset,
                                batch_size=batch_size,
                                shuffle=False)

    test_dataloader = DataLoader(dataset=testdataset,
                                batch_size=batch_size,
                                shuffle=False)

    if if_train:
        return {
            'train_dataloader':train_dataloader,
            'valid_dataloader':valid_dataloader,
        }
    
    else:
        return {
            'test_dataloader':test_dataloader
        }

    
if __name__ == "__main__":
    save_mapping()