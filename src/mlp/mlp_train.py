embedding_dim = 128
hidden_dim = 64
from pathlib import Path
import random
import numpy as np
import torch
import json

'''导入char_to_id label_to_id'''
mappings_path = Path(__file__).parent.parent.parent / "outputs" / "mappings"
char_to_id_path = mappings_path / "char_to_id.json"
label_to_id_path = mappings_path / "label_to_id.json"

with char_to_id_path.open('r',encoding='utf-8') as char_to_id_file:
    char_to_id = json.load(char_to_id_file)

with label_to_id_path.open('r',encoding='utf-8') as label_to_id_file:
    label_to_id = json.load(label_to_id_file)


def set_seed(seed: int = 15) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

#2.单轮训练
from .mlp_model import TicketClassifierMLP
def train_epoch(train_dataloader,model,loss_fn,opti):

    model.train() # 记得开dropout

    epoch_loss = 0.0
    num_correct = 0
    num = 0
    for batch in train_dataloader:
        opti.zero_grad()
        
        '''x:(batch_size,max_len)
           label(batch_size)'''
        x = batch['text_idx']
        label = batch['label_idx']

        # 预测
        y_pred = model(x)
        '''y_pred:(batch_size,num_classes)'''
        label_pred = y_pred.argmax(dim=1)
        '''class_pred:(batch_size)'''

        # 记录acc
        num_correct += (label == label_pred).sum().item() # 取消tensor格式
        num += x.size(0)

        # 记录loss
        loss = loss_fn(y_pred,label) # 这里是平均损失
        epoch_loss += loss.item() * x.size(0)

        # 更新参数
        loss.backward()
        opti.step()

    accuracy = num_correct / num
    avg_loss = epoch_loss / num # 避免样本数量的影响

    return {
        'avg_loss':avg_loss,
        'accuracy':accuracy
    }

#3.单轮验证
def valid_epoch(valid_dataloader,model,loss_fn):

    model.eval() # 记得关dropout

    epoch_loss = 0.0
    num_correct = 0
    num = 0
    all_label = []
    all_label_pred = []

    import torch
    with torch.no_grad():
        for batch in valid_dataloader:
            '''x:(batch_size,max_len)
            label(batch_size)'''
            x = batch['text_idx']
            label = batch['label_idx']
            all_label.extend(label.tolist())

            # 预测
            y_pred = model(x)
            '''y_pred:(batch_size,num_classes)'''
            label_pred = y_pred.argmax(dim=1)
            '''class_pred:(batch_size)'''
            all_label_pred.extend(label_pred.tolist())

            # 记录acc
            num_correct += (label == label_pred).sum().item()
            num += x.size(0)

            # 记录loss
            loss = loss_fn(y_pred,label) # 这里是平均损失
            epoch_loss += loss.item() * x.size(0)

    accuracy = num_correct / num
    avg_loss = epoch_loss / num # 避免样本数量的影响

    from sklearn.metrics import f1_score
    f1 = f1_score(y_true=all_label,y_pred=all_label_pred,
                  average='macro',zero_division=0) # 除数为0则直接f1=0
    
    return {
        'avg_loss':avg_loss,
        'accuracy':accuracy,
        'f1-score':f1
    }

#4.多轮验证
def train_epochs(data,dropout,lr,num_epoch,choice_name):
    set_seed(15)

    # 定义模型
    vocab_size = len(char_to_id)
    num_classes = len(label_to_id)

    model = TicketClassifierMLP(vocab_size=vocab_size,
                                embedding_dim=embedding_dim,
                                padding_id=char_to_id['<PAD>'],
                                hidden_dim=hidden_dim,
                                dropout=dropout,
                                num_classes=num_classes)

    import torch.nn as nn
    import torch
    loss_fn = nn.CrossEntropyLoss()
    opti = torch.optim.Adam(model.parameters(),lr = lr)

    history = {
        'train_loss':[],
        'train_acc':[],
        'valid_loss':[],
        'valid_acc':[],
        'valid_f1':[]
    }

    best_valid_loss = float('inf')   
    best_valid_acc = 0.0
    best_valid_f1 = float('-inf')
    best_epoch = 0 # 判断是否早停
    print(choice_name)
    for epoch in range(num_epoch):
        train_results =  train_epoch(train_dataloader=data['train_dataloader'],model=model,
                                     loss_fn=loss_fn,opti=opti)
        '''
        {
            'avg_loss':avg_loss,
            'accuracy':accuracy
        }每经历一次，模型就会更新一次'''

        valid_results = valid_epoch(valid_dataloader=data['valid_dataloader'],model=model,
                                    loss_fn=loss_fn)
        '''
        {
            'avg_loss':avg_loss,
            'accuracy':accuracy,
            'f1-score':f1
        }它模型不会变'''

        # 存个档
        history['train_loss'].append(train_results['avg_loss'])
        history['train_acc'].append(train_results['accuracy'])
        history['valid_loss'].append(valid_results['avg_loss'])
        history['valid_acc'].append(valid_results['accuracy'])
        history['valid_f1'].append(valid_results['f1-score'])

        #打个印
        print(
            f"Epoch {epoch + 1:02d}/{num_epoch} | "
            f"train loss: {train_results['avg_loss']:.4f} | "
            f"valid loss: {valid_results['avg_loss']:.4f} | "
            f"train acc: {train_results['accuracy']:.4f} | "
            f"valid acc: {valid_results['accuracy']:.4f} | "
            f"valid F1: {valid_results['f1-score']:.4f}"
        )

        if valid_results['f1-score'] > best_valid_f1:

            best_valid_loss = valid_results['avg_loss']
            best_valid_acc = valid_results['accuracy']
            best_valid_f1 = valid_results['f1-score']
            '''并不是找各个指标最优的，而是统计最佳模型的这仨指标'''
            best_epoch = epoch + 1

            model_save_path = Path(__file__).parent.parent.parent / 'outputs'/ 'mlp' / f'best_mlp_model_{choice_name}.pt'
            torch.save(model.state_dict(),model_save_path)
            '''第一次保存的并一定是最优的'''

    return {
        'params_conf':{
            'lr':lr,
            'dropout':dropout,
        },
        'history':history,
        'model_save_path':model_save_path,
        'best_epoch': best_epoch,        
        'best_valid_loss':best_valid_loss, 
        'best_valid_acc':best_valid_acc,
        'best_valid_f1' :best_valid_f1,
    }

# 选择不同的超参数组合来进行多轮训练
def train_with_different_params():
    #1.准备数据
    from ..tool.csv_to_dataloader import prepare_data
    data = prepare_data(if_train=True)

    choices = [
        {
            "choice_name": "A",
            "lr": 1e-3,
            "dropout": 0.0,
        },
        {
            "choice_name": "B",
            "lr": 1e-3,
            "dropout": 0.3,
        },
        {
            "choice_name": "C",
            "lr": 1e-2,
            "dropout": 0.3,
        },
    ]

    results = []
    num_epochs = 10
    for choice in choices:

        results.append(train_epochs(data=data,dropout=choice['dropout'],
                                    lr=choice['lr'],num_epoch=num_epochs,
                                    choice_name=choice['choice_name']))
        '''{
            'params_conf':{
                'lr':lr,
                'dropout':dropout,
            },
            'history':history,
            'model_save_path':model_save_path,
            'best_epoch': best_epoch,        
            'best_valid_loss':best_valid_loss, 
            'best_valid_acc':best_valid_acc,
            'best_valid_f1' :best_valid_f1,
        }'''

    best_result = max(results,key=lambda result : result['best_valid_f1'])

    '''接下来进入保存工作'''
    # 1.保存模型的配置
    best_model_conf = {
        'vocab_size':len(char_to_id),
        'embedding_dim':embedding_dim,
        'padding_id':char_to_id['<PAD>'],
        'hidden_dim':hidden_dim,
        'dropout':best_result['params_conf']['dropout'],
        'num_classes':len(label_to_id),
        'lr':best_result['params_conf']['lr']
    }
    '''model = TicketClassifierMLP(vocab_size=vocab_size,
                            embedding_dim=embedding_dim,
                            padding_id=data['char_to_id']['<PAD>'],
                            hidden_dim=hidden_dim,
                            dropout=dropout,
                            num_classes=num_classes)'''
    import json
    best_model_conf_path = Path(__file__).parent.parent.parent / "outputs" / 'mlp' / "best_mlp_model_config.json"
    with best_model_conf_path.open('w',encoding='utf-8') as best_model_file:   
        json.dump(
            best_model_conf,
            best_model_file,
            ensure_ascii=False, # 写中文的格式
            indent=2,# 缩进2格
        )

    #2.保存history
    history_path = Path(__file__).parent.parent.parent / "outputs" / 'mlp' / "mlp_history.json"
    with history_path.open('w',encoding='utf-8') as history_file:   
        json.dump(
            best_result['history'],
            history_file,
            ensure_ascii=False, # 写中文的格式
            indent=2,# 缩进2格
        )

    #3.保存最佳模型，采用路径复制
    best_model_path = Path(__file__).parent.parent.parent / 'outputs' / 'mlp' / 'best_mlp_model.pt'
    import shutil
    shutil.copyfile(best_result['model_save_path'],best_model_path)

    #4.保存最优指标
    best_index_path = Path(__file__).parent.parent.parent / 'outputs' / 'mlp' / 'best_mlp_model_index.json'
    with best_index_path.open('w',encoding='utf-8') as best_index_file:   
        json.dump(
            {
                'best_epoch':best_result['best_epoch'],        
                'best_valid_loss':best_result['best_valid_loss'], 
                'best_valid_acc':best_result['best_valid_acc'],
                'best_valid_f1' :best_result['best_valid_f1'],
            },
            best_index_file,
            ensure_ascii=False, # 写中文的格式
            indent=2,# 缩进2格
        )


if __name__ == '__main__':
    train_with_different_params()