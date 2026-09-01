from pathlib import Path
import json
import torch
from functools import lru_cache


def encode_text(text, char_to_id, max_len):
    text = text[:max_len]

    encoded = [
        char_to_id.get(char, char_to_id["<UNK>"])
        for char in text
    ]

    padding_size = max_len - len(encoded)
    encoded.extend(
        [char_to_id["<PAD>"]] * padding_size
    )

    return encoded


'''路径设置'''
project_path = Path(__file__).parent.parent
outputs_root = project_path / 'outputs' 
max_len_path = outputs_root / 'decide_max_len' / 'max_len.json'

'''转化表'''
mappings_path = Path(__file__).parent.parent / "outputs" / "mappings"
char_to_id_path = mappings_path / "char_to_id.json"
label_to_id_path = mappings_path / "label_to_id.json"

with char_to_id_path.open('r',encoding='utf-8') as char_to_id_file:
    char_to_id = json.load(char_to_id_file)

with label_to_id_path.open('r',encoding='utf-8') as label_to_id_file:
    label_to_id = json.load(label_to_id_file)

id_to_label = {id:label for label,id in label_to_id.items()}

'''加载max_len等参数'''
with max_len_path.open('r',encoding='utf-8') as file:
    max_len = json.load(file)['max_len']


'''不要一开始加载所有的模型,而是懒加载
第一次调用：从磁盘读取模型
第二次调用：直接返回内存中的模型
第三次调用：继续返回内存中的模型
因此不会每次预测都重新读取文件'''
@lru_cache(maxsize=1)
def load_baseline():
    import joblib

    '''加载baseline模型和tfidf编码器'''
    baseline_model_path = outputs_root / 'baseline' /"model.joblib"
    baseline_tfidf_path = outputs_root / 'baseline' / "tfidf.joblib"
    if not baseline_tfidf_path.exists():
        raise FileNotFoundError(f'找不到TF-IDF分词文件：{baseline_tfidf_path}')

    if not baseline_model_path.exists():
        raise FileNotFoundError(f'找不到模型文件：{baseline_model_path}')
    
    baseline_model = joblib.load(baseline_model_path)
    baseline_tfidf = joblib.load(baseline_tfidf_path)

    return baseline_tfidf,baseline_model


def predict_baseline(text):
    tfidf,model = load_baseline()

    features = tfidf.transform([text]) #baseline_test里面模型接受的就是编码后的list
    '''features:(1,num_features)'''

    predictions = model.predict(features)
    '''类别标签，一维NumPy数组，只有一个数,array([2])'''

    probability = model.predict_proba(features) 
    '''预测每个类别的概率，一维NumPy数组(1,num_classes)'''

    max_probability = probability.max(axis=1) # 找每行最大的列，就是对应类别的概率
    '''类别概率，一维NumPy数组(1,)'''

    return {
        'model':'TF-IDF + LR',
        'predictions':predictions.item(),
        'max_probability':max_probability.item()
    }


'''懒加载mlp模型'''
@lru_cache(maxsize=1)
def load_mlp():
    from ..src.mlp.mlp_model import TicketClassifierMLP

    '''加载mlp模型'''
    best_mlp_model_path = outputs_root / 'mlp' /'best_mlp_model.pt'
    best_mlp_model_conf_path = outputs_root / 'mlp' / 'best_mlp_model_config.json'

    if not best_mlp_model_path.exists():
        raise FileNotFoundError(f'找不到mlp模型文件：{best_mlp_model_path}')

    if not best_mlp_model_conf_path.exists():
        raise FileNotFoundError(f'找不到mlp模型配置文件：{best_mlp_model_conf_path}')

    with best_mlp_model_conf_path.open('r',encoding='utf-8') as best_mlp_model_conf_file:
        best_mlp_model_conf = json.load(best_mlp_model_conf_file)

    best_mlp_model = TicketClassifierMLP(vocab_size=best_mlp_model_conf['vocab_size'],
                                        embedding_dim=best_mlp_model_conf['embedding_dim'],
                                        padding_id=best_mlp_model_conf['padding_id'],
                                        hidden_dim=best_mlp_model_conf['hidden_dim'],
                                        dropout=best_mlp_model_conf['dropout'],
                                        num_classes=best_mlp_model_conf['num_classes'])
    mlp_dict = torch.load(best_mlp_model_path)
    best_mlp_model.load_state_dict(mlp_dict)

    return best_mlp_model


def predict_mlp(text):
    model = load_mlp()
    model.eval()

    with torch.no_grad():

        '''先编码，这是原来dataset的流程'''
        encoded = encode_text(text,char_to_id,max_len)
        encoded = torch.tensor(encoded, dtype=torch.long).unsqueeze(0)
        '''模型需要tensor(batch_size,max_len)的尺寸'''  

        y = model(encoded) #tensor(1,num_class)
        probability = torch.softmax(y,dim=1) #tensor(1,num_class)
        max_probability,labels_pred = probability.max(dim=1) #tensor(1,1),tensor(1,1)

    return {
        'model':'MLP',
        'predictions':id_to_label[labels_pred.item()],
        'max_probability':max_probability.item()
    }


'''懒加载bert模型'''
@lru_cache(maxsize=1)
def load_bert():

    from transformers import AutoTokenizer
    from transformers import AutoModelForSequenceClassification

    best_BERT_model_path = outputs_root / 'BERT' / 'best_BERT_model.pt'
    best_BERT_model_conf_path = outputs_root / 'BERT' / 'best_BERT_model_config.json'

    if not best_BERT_model_path.exists():
        raise FileNotFoundError(f'找不到bert模型文件：{best_BERT_model_path}')

    if not best_BERT_model_conf_path.exists():
        raise FileNotFoundError(f'找不到bert模型配置文件：{best_BERT_model_conf_path}')

    with best_BERT_model_conf_path.open('r',encoding='utf-8') as best_BERT_model_conf_file:
        best_BERT_model_conf = json.load(best_BERT_model_conf_file)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 分词器
    tokenizer = AutoTokenizer.from_pretrained(best_BERT_model_conf['model_name'])

    # 模型
    best_BERT_model = AutoModelForSequenceClassification.from_pretrained(
        best_BERT_model_conf['model_name'],
        num_labels=len(label_to_id),
    ).to(device)

    BERT_model_dict = torch.load(best_BERT_model_path,map_location=device,weights_only=True)
    best_BERT_model.load_state_dict(BERT_model_dict)

    return best_BERT_model,tokenizer,device


def predict_BERT(text):
    model,tokenizer,device = load_bert()
    model.eval()
    with torch.no_grad():

        '''先编码，这是原来dataset的流程'''
        encoded = tokenizer(
            text,                 # 把这些文本进行编码
            padding='max_length',          # 短句补 PAD
            truncation=True,       # 长句截断
            return_tensors="pt",   # 转成 PyTorch Tensor
            max_length = max_len
        )
        '''包含编码后的input_ids,padding_mask,label_ids,
        尺寸均为(batch_size,max_len)'''

        outputs = model(input_ids=encoded['input_ids'].to(device),
                        attention_mask=encoded['attention_mask'].to(device))
        
        logits = outputs.logits #tensor(1,num_labels)
        probability = torch.softmax(logits,dim=1)
        max_probability,labels_pred = probability.max(dim=1) # tensor(1,1)

    return {
        'model':'BERT',
        'predictions':id_to_label[labels_pred.item()],
        'max_probability':max_probability.item()
    }


def predict(text, model_name):

    if model_name == "tfidf_lr":
        return predict_baseline(text)

    if model_name == "mlp":
        return predict_mlp(text)

    if model_name == "bert":
        return predict_BERT(text)

    raise ValueError(f"Unknown model: {model_name}")
'''uvicorn 启动
        ↓
找到 app/main.py
        ↓
执行 main.py
        ↓
main.py import model_service
        ↓
Python 执行 model_service.py
        ↓
加载 MLP / TF-IDF+LR / BERT
        ↓
FastAPI 启动完成
        ↓
    等待请求'''
