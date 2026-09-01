from pathlib import Path
import matplotlib.pyplot as plt

# 让 Matplotlib 显示中文
plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
]
# 避免负号显示异常
plt.rcParams["axes.unicode_minus"] = False

#1. 准备数据
project_path = Path(__file__).parent.parent.parent
outputs_path = project_path / 'outputs' / 'baseline'
outputs_path.mkdir(parents=True,exist_ok=True)
test_path = project_path / 'data' / 'test.csv'
model_path = outputs_path / "model.joblib"
tfidf_path = outputs_path / "tfidf.joblib"
png_save_path = outputs_path / 'confusion_matrix.png'
error_file_path = outputs_path / 'error_file.csv'

import pandas as pd
import joblib
def prepare_data_model(test_path,model_path,tfidf_path):

    test_file = pd.read_csv(test_path)

    model = joblib.load(model_path)
    tfidf = joblib.load(tfidf_path)

    '''train的prepare_data是返回list形式的train_texts/labels,valid_texts/labels
    这里的prepare_data_model是返回dataframe形式的test_file,以及model和tfidf'''
    return {
        "test_file":test_file,
        'model':model,
        'tfidf':tfidf
    }


def build_tfidf(data):

    test_texts = data['test_file']['text'].tolist()
    test_labels = data['test_file']['label'].tolist()
    tfidf = data['tfidf']
    model = data['model']

    # tfidf中的词表已经在训练集中训练好，这里只需测试集上编码就行
    '''test_features:(len(test_texts),num_features)'''
    test_features = tfidf.transform(test_texts)

    '''把test_file拆成texts，labels。texts编码为features'''
    return {
        "test_file":data['test_file'],
        "test_features": test_features,
        'test_labels':test_labels,
        'model':model,
        "tfidf": tfidf
    }


from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix
)
#绘制混淆矩阵
def plot_confusion_matrix(labels,predictions,class_names,save_path):
    print(labels[0])
    print(predictions[0]) # 都是同类型的就可以画图
    """
    混淆矩阵：

    行：真实类别
    列：预测类别
    """
    matrix = confusion_matrix(y_true=labels,y_pred=predictions,labels=class_names)

    display = ConfusionMatrixDisplay(confusion_matrix=matrix,display_labels=class_names)

    display.plot(values_format="d", xticks_rotation=30,)

    plt.xlabel("预测类别")
    plt.ylabel("真实类别")
    plt.title("TF-IDF + Logistic Regression 测试集混淆矩阵")
    plt.tight_layout()
    plt.savefig(save_path,dpi=300)
    plt.close()


from common.validuate_f import evaluate
def run_validuate(data): # 只负责得到指标

    model = data['model']
    test_features = data['test_features']
    test_labels = data['test_labels']

    predictions = model.predict(test_features) # 预测结果类别，文字
    '''predictions是保存字符串类别标签的一维NumPy数组'''

    probability = model.predict_proba(test_features) #预测每个类别的概率
    max_probability = probability.max(axis=1) # 找每行最大的列，就是对应类别的概率

    '''调用train.py中的评价函数，只不过评价集从valid到test'''
    metrics = evaluate(test_labels,predictions)
    '''{
        "accuracy": round(float(accuracy),4),
        "precision_macro": round(float(precision_macro),4),
        "recall_macro": round(float(recall_macro),4),
        "f1_macro": round(float(f1_macro),4),
        "per_class": class_metrics
    }'''

    # results
    return {
        'metrics':metrics,
        'predictions':predictions,
        'max_probability':max_probability
    }


def save_results(results):
    #保存测试集指标
    import json
    test_metrics_path = outputs_path / 'test_metrics.json'
    metrics = results['metrics']
    with test_metrics_path.open('w',encoding='utf-8') as test_metrics_file:   
         json.dump(
            metrics,
            test_metrics_file,
            ensure_ascii=False, # 写中文的格式
            indent=2,# 缩进2格
         )


def get_possible_reason(text,true_label,predicted_label,confidence,):
    """
    根据错误类别和文本特征，
    给出一个初步的错误原因。

    注意：
    这里只是规则推测，
    不是模型自动理解了错误原因。
    """

    label_pair = {true_label,predicted_label}

    if label_pair == {"网络故障","信号问题"}:
        return (
            "网络故障与信号问题存在相似词，"
            "例如网络、断开、无法连接"
        )
    
    if label_pair == {"套餐业务","费用问题"}:
        return (
            "文本同时涉及套餐和收费，"
            "两个类别的关键词发生重叠"
        )

    if (true_label == "投诉建议" and predicted_label != "投诉建议"):
        return (
            "投诉文本中包含具体业务词，"
            "模型更关注业务关键词，"
            "没有充分识别投诉意图"
        )

    if (predicted_label == "投诉建议" and true_label != "投诉建议"):
        return (
            "文本包含处理、客服、维修等词，"
            "模型将业务问题误认为投诉"
        )

    negative_words = [
        "没有",
        "不是",
        "不能",
        "不需要",
        "未办理",
        "没开通",
        "取消",
    ]

    for word in negative_words:
        if word in text:
            return (
                "文本包含否定表达，"
                "TF-IDF 无法充分理解否定关系"
            )

    if confidence < 0.60:
        return (
            "模型置信度较低，"
            "可能存在标签边界模糊或表达不典型"
        )

    return (
        "可能受到高频关键词影响，"
        "TF-IDF无法充分理解上下文和完整意图"
    )


# 保存错误案例
def save_error_cases(test_file,predictions,confidence,save_path):
    """
    保存预测错误的测试数据。构建dataframe->csv
    字段：text true_label predicted_label confidence possible_reason"""
    result_file = pd.DataFrame(
        {
            'text': test_file['text'].tolist(),
            'true_label':test_file['label'].tolist(),
            'predicted_label':predictions,
            'confidence':confidence
        }
    )

    # copy至期待的dataframe
    error_file = result_file[result_file['true_label'] != result_file['predicted_label']].copy()

    if error_file.empty:
        error_file["possible_reason"] = pd.Series(dtype="object")
        error_file.to_csv(save_path,index=False,encoding="utf-8-sig")
        print("测试集中没有预测错误的样本。")
        return

    error_file['confidence'] = error_file['confidence'].round(4)
    error_file['possible_reason'] = error_file.apply(
        lambda row : get_possible_reason(text=row['text'],true_label=row['true_label'],
                                         predicted_label=row['predicted_label'],
                                         confidence=row['confidence']),
        axis = 1
    )
    '''apply(axis=1) 逐行读取错误样本，把每一行的信息传给 get_possible_reason()，
        并将函数返回值保存到 possible_reason 列'''

    error_file = (error_file.sort_values(by="confidence",ascending=False)
         .reset_index(drop=True)) #ascending=False = reverse=True

    error_file.to_csv(save_path,index=False,
                              encoding='utf-8')


def main():
    data = prepare_data_model(test_path,model_path,tfidf_path)
    '''{
        "test_file":test_file,dataframe形式的
        'model':model,
        'tfidf':tfidf
    }'''

    data = build_tfidf(data)
    '''{
        "test_file":data['test_file'],
        "test_features": test_features,
        'test_labels':test_labels,
        'model':model,
        "tfidf": tfidf
    }'''

    results = run_validuate(data)
    '''
    {
        'metrics':{
                    "accuracy": round(float(accuracy),4),
                    "precision_macro": round(float(precision_macro),4),
                    "recall_macro": round(float(recall_macro),4),
                    "f1_macro": round(float(f1_macro),4),
                    "per_class": class_metrics
                }
        'predictions':predictions,
        'max_probability':max_probability
    }
    '''

    # 画混淆矩阵
    class_names = data['model'].classes_.tolist()
    plot_confusion_matrix(labels=data['test_labels'],predictions=results['predictions'],
                          class_names=class_names,save_path=png_save_path)


    save_results(results)

    save_error_cases(test_file=data['test_file'],predictions=results['predictions'],
                     confidence=results['max_probability'],save_path=error_file_path)


if __name__ == '__main__':
    main()
    print('test done')