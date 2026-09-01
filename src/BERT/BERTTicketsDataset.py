import torch
import torch.nn as nn

class BERTTicketsDataset(torch.utils.data.Dataset):
    def __init__(self,texts,labels,tokenizer,max_len):
        super().__init__()
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer # Hugging Face的 AutoTokenizer
        self.max_len = max_len


    def __len__(self):
        return len(self.texts)


    def __getitem__(self, index):
        text = self.texts[index]
        label = self.labels[index]
        encoded = self.tokenizer(
            text,                 # 把这些文本进行编码
            padding='max_length',          # 短句补 PAD
            truncation=True,       # 长句截断
            return_tensors="pt",   # 转成 PyTorch Tensor
            max_length = self.max_len
        )

        '''encoded['input_ids']：(batch_size,man_len)只不过这里是batch_size=1
        需要删除第一维，剩下(max_len)，让DataLoader扩展为(batch_size,man_len)'''

        return {
            'input_ids':encoded['input_ids'].squeeze(0), #tensor(max_len)
            'label_ids':torch.tensor(label,dtype=torch.long), #tensor(1)
            'padding_mask':encoded['attention_mask'].squeeze(0) #tensor(max_len)
        }

        
    
        