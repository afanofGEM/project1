from pathlib import Path
import random
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification
from sklearn.metrics import f1_score
from transformers import get_linear_schedule_with_warmup
from .BERTTicketsDataLoader import prepare_data
import json
import shutil

MODEL_NAME = "google-bert/bert-base-chinese"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
project_root = Path(__file__).parent.parent.parent
output_root = project_root / 'outputs'/ 'BERT'

max_grad_norm = 1.0
num_epochs = 10

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


def train_epoch(train_dataloader,model,opti,scheduler):

    model.train()

    loss_epoch = 0.0
    accuracy_epoch = 0.0
    correct_num = 0
    num = 0
    for batch in train_dataloader:
        opti.zero_grad()

        x = batch['input_ids'].to(device) # batch_size,max_len
        labels = batch['label_ids'].to(device) # batch_size
        padding_mask = batch['padding_mask'].to(device) # batch_size,max_len

        outputs = model(input_ids=x,attention_mask=padding_mask,labels=labels)
        logits = outputs.logits #batch_size,num_labels由model定
        labels_pred = logits.argmax(dim=1) # batch_size
        loss = outputs.loss

        num += x.size(0)
        correct_num += (labels_pred == labels).sum().item()
        loss_epoch += loss.item() * x.size(0)

        loss.backward()
        '''防止梯度过大'''
        torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=max_grad_norm)

        opti.step()
        '''更新学习率learning rate'''
        scheduler.step()

    accuracy_epoch = correct_num / num
    avg_loss_epoch = loss_epoch / num

    return {
        'avg_loss_epoch':avg_loss_epoch,
        'accuracy_epoch':accuracy_epoch
    }


def valid_epoch(valid_dataloader,model):
    model.eval()

    loss_epoch = 0.0
    accuracy_epoch = 0.0
    correct_num = 0
    num = 0
    all_labels = []
    all_labels_pred = []
    with torch.no_grad():
        for batch in valid_dataloader:

            x = batch['input_ids'].to(device) # batch_size,max_len
            labels = batch['label_ids'].to(device) # batch_size
            all_labels.extend(labels.cpu().tolist())
            padding_mask = batch['padding_mask'].to(device) # batch_size,max_len
    
            outputs = model(input_ids=x,attention_mask=padding_mask,labels=labels)
            logits = outputs.logits #batch_size,num_labels由model定
            labels_pred = logits.argmax(dim=1) # batch_size
            all_labels_pred.extend(labels_pred.cpu().tolist()) # 要把tensor转化成list
            loss = outputs.loss
    
            num += x.size(0)
            correct_num += (labels_pred == labels).sum().item()
            loss_epoch += loss.item() * x.size(0)

    accuracy_epoch = correct_num / num
    avg_loss_epoch = loss_epoch / num
    f1 = f1_score(y_true=all_labels,y_pred=all_labels_pred,average='macro',zero_division=0) # 除数为0则直接f1=0

    return {
        'avg_loss_epoch':avg_loss_epoch,
        'accuracy_epoch':accuracy_epoch,
        'f1-score':f1
    }


'''权重衰减w = w-lr*gradient-lr*weight_decay*w再往变化的反方向拉一点，防止过拟合'''
def train_epochs(data,lr,weight_decay,warmup_ratio,num_epoch,choice_name):

    train_dataloader = data['train_dataloader']
    valid_dataloader = data['valid_dataloader']

    model = AutoModelForSequenceClassification.from_pretrained(
      MODEL_NAME,
      num_labels=len(label_to_id)
   ).to(device)

    opti = torch.optim.AdamW(
      model.parameters(),
      lr=lr,
      weight_decay=weight_decay
   )

    '''相比于mlp多的初始化步骤:设置scheduler'''
    num_training_steps = len(data['train_dataloader']) * num_epoch
    num_warmup_steps = int(num_training_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        opti,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    '''scheduler调度器控制lr
    学习率的变化包括两个阶段：
    Learning Rate Warmup（学习率预热）和 Linear Learning Rate Decay（线性学习率衰减）'''

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
        train_results =  train_epoch(train_dataloader=train_dataloader,model=model,
                                     opti=opti,scheduler=scheduler)
        '''
        {
            'avg_loss_epoch':avg_loss_epoch,
            'accuracy_epoch':accuracy_epoch
        }每经历一次，模型就会更新一次'''

        valid_results = valid_epoch(valid_dataloader=valid_dataloader,model=model)
        '''
        {
            'avg_loss_epoch':avg_loss_epoch,
            'accuracy_epoch':accuracy_epoch,
            'f1-score':f1
        }它模型不会变'''

        # 存个档
        history['train_loss'].append(train_results['avg_loss_epoch'])
        history['train_acc'].append(train_results['accuracy_epoch'])
        history['valid_loss'].append(valid_results['avg_loss_epoch'])
        history['valid_acc'].append(valid_results['accuracy_epoch'])
        history['valid_f1'].append(valid_results['f1-score'])

        #打个印
        print(
            f"Epoch {epoch + 1:02d}/{num_epoch} | "
            f"train loss: {train_results['avg_loss_epoch']:.4f} | "
            f"valid loss: {valid_results['avg_loss_epoch']:.4f} | "
            f"train acc: {train_results['accuracy_epoch']:.4f} | "
            f"valid acc: {valid_results['accuracy_epoch']:.4f} | "
            f"valid F1: {valid_results['f1-score']:.4f}"
        )

        if valid_results['f1-score'] > best_valid_f1:
            best_choice_name = choice_name
            best_valid_loss = valid_results['avg_loss_epoch']
            best_valid_acc = valid_results['accuracy_epoch']
            best_valid_f1 = valid_results['f1-score']
            '''并不是找各个指标最优的，而是统计最佳模型的这仨指标'''
            best_epoch = epoch + 1

            model_save_path = output_root / f'best_BERT_model_{choice_name}.pt'
            torch.save(model.state_dict(),model_save_path)
            '''第一次保存的并一定是最优的'''

    return {
        'params_conf':{
            'lr':lr,
            'weight_decay':weight_decay,
            'warmup_ratio':warmup_ratio,
            'best_choice_name':best_choice_name,
            'model_name':MODEL_NAME,
            'num_labels':len(label_to_id)
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
    data = prepare_data(if_train=True)

    choices = [
        {
            'choice_name':'A',
            'lr':2e-5,
            'weight_decay':0.01,
            'warmup_ratio':0.1
        },
        {
            'choice_name':'B',
            'lr':3e-5,
            'weight_decay':0.01,
            'warmup_ratio':0.1
        },
        {
            'choice_name':'C',
            'lr':5e-5,
            'weight_decay':0.01,
            'warmup_ratio':0.1
        },
        {
            'choice_name':'D',
            'lr':3e-5,
            'weight_decay':0.0,
            'warmup_ratio':0.1
        }
    ]


    results = []

    for choice in choices:

        results.append(train_epochs(data=data,
                                    lr=choice['lr'],num_epoch=num_epochs,
                                    weight_decay=choice['weight_decay'],
                                    warmup_ratio=choice['warmup_ratio'],
                                    choice_name=choice['choice_name']))
        '''{
            'params_conf':{
                'lr':lr,
                'weight_decay':weight_decay,
                'warmup_ratio':warmup_ratio,
                'best_choice_name':best_choice_name,
                'model_name':MODEL_NAME,
                'num_labels':len(label_to_id)
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
    best_model_conf_path = output_root/ "best_BERT_model_config.json"
    with best_model_conf_path.open('w',encoding='utf-8') as best_model_conf_file:   
        json.dump(
            best_result['params_conf'],
            best_model_conf_file,
            ensure_ascii=False, # 写中文的格式
            indent=2,# 缩进2格
        )

    #2.保存history
    history_path = output_root / "BERT_history.json"
    with history_path.open('w',encoding='utf-8') as history_file:   
        json.dump(
            best_result['history'],
            history_file,
            ensure_ascii=False, # 写中文的格式
            indent=2,# 缩进2格
        )

    #3.保存最佳模型，采用路径复制
    best_model_path = output_root / 'best_BERT_model.pt'
    shutil.copyfile(best_result['model_save_path'],best_model_path)

    #4.保存最优指标
    best_index_path = output_root / 'best_BERT_model_index.json'
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
    set_seed()
    train_with_different_params()