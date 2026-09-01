import torch.nn as nn

class TicketClassifierMLP(nn.Module):

    def __init__(self,vocab_size,embedding_dim,padding_id,hidden_dim,dropout,num_classes):

        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                      embedding_dim=embedding_dim,
                                      padding_idx=padding_id)
        self.padding_id = padding_id

        self.mlp = nn.Sequential(
            nn.Linear(in_features=embedding_dim,out_features=hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=hidden_dim,out_features=num_classes)
        )


    def forward(self,text_ids):
        '''text_ids:(batch_size,max_len)是从DataLoader取出来的
        [
            [1,2,3],
            [0,5,6]
        ]
        '''

        embedded = self.embedding(text_ids) 
        '''(batch_size,max_len,embedding_dim)
        [
            [[1,1,1],[2,2,2],[3,3,3]],
            [[0.1,0.1,0.1],[5,5,5],[6,6,6]]
        ]'''

        # 便于和embedded预算，同维度
        judge_padding = text_ids.ne(self.padding_id).unsqueeze(-1)
        '''       
        [
            [T,T,T],
            [F,T,T]
        ] (batch_size,max_len)

        [
            [[T],[T],[T]],
            [[F],[T],[T]]
        ] (batch_size,max_len,1)
        '''

        # 把原先embedding padding的部分全部置0
        embedded = embedded * judge_padding 
        '''(batch_size,max_len,embedding_dim)
        [
            [[1,1,1],[2,2,2],[3,3,3]],
            [[0,0,0],[5,5,5],[6,6,6]]
        ]'''

        # 平均池化1 先把embedding_dim合并
        embedded = embedded.sum(dim=1) # 消除max_len维度
        '''(batch_size,embedding_dim)
        [
            [6,6,6],
            [11,11,11]
        ]'''

        # 平均池化2 求每条文本的有效长度
        count_num = judge_padding.sum(dim=1).clamp(min=1) # 如果全是padding记为1
        ''' (batch_size,1)     
        [
            [3],
            [2]
        ]'''

        embedded = embedded / count_num
        '''(batch_size,embedding_dim)
        [
            [2,2,2],
            [5.5,5.5,5.5]
        ]'''

        outputs = self.mlp(embedded)
        '''(batch_size,num_classes)'''
        return outputs


if __name__ == '__main__':
    vocab_size = 1000
    embedding_dim = 64
    padding_id = 0
    hidden_dim = 128
    num_classes = 5
    dropout = 0.3
    model = TicketClassifierMLP(vocab_size=vocab_size,embedding_dim=embedding_dim,
                           padding_id=padding_id,hidden_dim=hidden_dim,
                           dropout=dropout,num_classes=num_classes)
    
    import torch
    text_ids = torch.randint(low=1,high=vocab_size,size=(8,32))
    text_ids[0, 10:] = padding_id
    text_ids[1, 15:] = padding_id
    text_ids[2, 20:] = padding_id
    text_ids[3, 8:] = padding_id
    result = model(text_ids)
    print("text_ids shape:", text_ids.shape)
    print("logits shape:", result.shape)
