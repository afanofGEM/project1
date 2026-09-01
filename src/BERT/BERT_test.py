from pathlib import Path
import torch
from .BERTTicketsDataLoader import prepare_data
import json
from common.validuate_f import evaluate
from common.validuate_f import plot_confusion_matrix
import pandas as pd
from transformers import AutoModelForSequenceClassification

MODEL_NAME = "google-bert/bert-base-chinese"
   
project_root = Path(__file__).parent.parent.parent
output_root = project_root / 'outputs'/ 'BERT'
data_path = project_root / 'data' / 'test.csv'
best_model_path = output_root / 'best_BERT_model.pt'
best_model_conf_path = output_root / 'best_BERT_model_config.json'
metrics_save_path = output_root / 'BERT_test_metrics.json'
img_save_path = output_root / 'confusion_matrix.png'
prediction_save_path = output_root / 'test_report.csv'
error_save_path = output_root / 'error_report.csv'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

'''导入char_to_id label_to_id'''
mappings_path = Path(__file__).parent.parent.parent / "outputs" / "mappings"
char_to_id_path = mappings_path / "char_to_id.json"
label_to_id_path = mappings_path / "label_to_id.json"

with char_to_id_path.open('r',encoding='utf-8') as char_to_id_file:
    char_to_id = json.load(char_to_id_file)

with label_to_id_path.open('r',encoding='utf-8') as label_to_id_file:
    label_to_id = json.load(label_to_id_file)


def run_evaluate(test_dataloader,model):
    model.eval()

    all_labels = []
    all_predictions = []
    all_confidence = []
    with torch.no_grad():
        for batch in test_dataloader:
            x = batch['input_ids'].to(device) #(batch_size,max_len)
            labels = batch['label_ids'].to(device) #(batch_size)
            padding_mask = batch['padding_mask'].to(device) # batch_size,max_len

            outputs = model(input_ids=x,attention_mask=padding_mask,labels=labels)
            logits = outputs.logits #batch_size,num_labels由model定

            '''先softmax再求最大概率'''
            probability = torch.softmax(logits,dim=1)
            max_probability,labels_pred = probability.max(dim=1) # batch_size

            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(labels_pred.cpu().tolist())
            all_confidence.extend(max_probability.cpu().tolist())


    metrics = evaluate(all_labels,all_predictions)
    '''{
        "accuracy": round(float(accuracy),4),
        "precision_macro": round(float(precision_macro),4),
        "recall_macro": round(float(recall_macro),4),
        "f1_macro": round(float(f1_macro),4),
        "per_class": class_metrics
    }'''

    return {
        'metrics':metrics,
        'labels':all_labels,
        'predictions':all_predictions,
        'confidence':all_confidence
    }


def save_metrics(metrics):
    with metrics_save_path.open('w',encoding='utf-8') as metrics_save_file:   
            json.dump(
            metrics,
            metrics_save_file,
            ensure_ascii=False, # 写中文的格式
            indent=2,# 缩进2格
        )


def save_predictions(test_file,predictions,confidence,id_to_label):
    prediction_file = pd.DataFrame(
        {
            "ticket_id":test_file["ticket_id"],
            "text":test_file["text"],
            "true_label":test_file["label"],
            "prediction":[id_to_label[prediction_id]
                          for prediction_id in predictions],
            "confidence":[round(value,4) for value in confidence],
        }
    )

    prediction_file.to_csv(
        prediction_save_path,
        index=False,
        encoding="utf-8-sig",
    )


def save_errors(test_file,predictions,confidence,id_to_label):
    
    error_file = test_file.loc[: , ["ticket_id","text","label"]]
    error_file['predictions'] = [id_to_label[pred] for pred in predictions]
    error_file['confidence'] = [round(con,4) for con in confidence]

    error_file = error_file.loc[error_file['predictions']!=error_file['label']]

    '''inplace=True表示直接修改原来的 DataFrame，不需要接收返回值'''
    error_file.rename(columns={"label": "true_label"},inplace=True)

    error_file.to_csv(  
        error_save_path,
        index=False,
        encoding="utf-8-sig",
    )


def main():

    # 1.准备数据
    data = prepare_data(if_train=False)
    '''{
            'test_dataloader':test_dataloader,
            'label_to_id':label_to_id
        }'''
    test_dataloader = data['test_dataloader']
    label_to_id = label_to_id

    # 2.准备模型
    with best_model_conf_path.open('r',encoding='utf-8') as best_model_conf_file:
        best_model_conf = json.load(best_model_conf_file)

    best_model = AutoModelForSequenceClassification.from_pretrained(
      best_model_conf['model_name'],
      num_labels=len(label_to_id),
   ).to(device)

    model_dict = torch.load(best_model_path,map_location=device,weights_only=True)
    '''用模型接受参数'''
    best_model.load_state_dict(model_dict)

    metrics_dict = run_evaluate(test_dataloader,best_model)
    '''{
        'metrics':metrics,
        'labels':all_labels,
        'predictions':all_predictions,
        'max_probability':max_probability
    }'''

    save_metrics(metrics_dict['metrics'])

    class_name = [name for name,_ in 
                   sorted(label_to_id.items(),key=lambda item: item[1])]
    
    plot_confusion_matrix(metrics_dict['labels'],metrics_dict['predictions'],
                          class_name,img_save_path)

    test_file = pd.read_csv(data_path)

    id_to_label = {id:label for label,id in label_to_id.items()}
    save_predictions(test_file,metrics_dict['predictions'],
                     metrics_dict['confidence'],id_to_label)

    save_errors(test_file,metrics_dict['predictions'],
                     metrics_dict['confidence'],id_to_label)


if __name__ == "__main__":
    main()