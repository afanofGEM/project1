from pathlib import Path

#1. 准备数据
project_path = Path(__file__).parent.parent.parent
data_path = project_path / 'data' / 'data.csv'
outputs_path = project_path / 'outputs' / 'baseline'
outputs_path.mkdir(parents=True,exist_ok=True)
train_path = project_path / 'data' / 'train.csv'
valid_path = project_path / 'data' / 'valid.csv'
test_path = project_path / 'data' / 'test.csv'

import pandas as pd
def prepare_data(train_path,valid_path,test_path):
    train_file = pd.read_csv(train_path)
    valid_file = pd.read_csv(valid_path)
    test_file = pd.read_csv(test_path)

    train_texts = train_file["text"].tolist()
    train_labels = train_file["label"].tolist()
    valid_texts = valid_file["text"].tolist()
    valid_labels = valid_file["label"].tolist()
    test_texts = test_file['text'].tolist()
    test_labels = test_file['label'].tolist()

    return {
        "train_texts": train_texts,
        "train_labels": train_labels,
        "valid_texts": valid_texts,
        "valid_labels": valid_labels,
        "test_texts":test_texts,
        "test_labels":test_labels
    }


# 4.训练过程,输入的data而不是路径
def run_experiment(data,max_iter,random_state):
    '''data:
    {
        "train_texts": train_texts,
        "train_labels": train_labels,
        "valid_texts": valid_texts,
        "valid_labels": valid_labels,
        "test_texts":test_texts,
        "test_labels":test_labels
    }'''
    # GridSearchCV 参数网络 搜索 交叉验证：通过训练和验证集来搜索参数与超参数的工具

    # 1.GridSearCV的数据准备部分
    # 1.1 GridSearchCV需要完整的训练集+验证集
    search_texts = data['train_texts'] + data['valid_texts']
    search_labels = data['train_labels'] + data['valid_labels']

    # 1.2 GridSearchCV需要通过索引告诉它哪些是训练集，哪些是验证集
    index = [-1] * len(data['train_texts']) + [0] * len(data['valid_texts'])
    from sklearn.model_selection import GridSearchCV,PredefinedSplit
    predefined_split = PredefinedSplit(index)

    # 2.GridSearchCV的工作流定义部分
    # 2.1 创建Pipeline
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    pipeline = Pipeline(steps=[('tfidf',TfidfVectorizer(analyzer='char',
                                                        ngram_range=(1,2))),
                               ('model',LogisticRegression(
                                                            max_iter=max_iter,
                                                            solver="lbfgs",
                                                            random_state=random_state))])
    '''lbfgs 会不断寻找让损失函数更小的参数，是 scikit-learn 逻辑回归的默认求解器'''

    # 3.GridSearchCV的参数范围
    # 因为LogisticRegression位于Pipeline的model步骤中，所以参数名称需要写成model__参数名
    params_range = {
        'model__C':[0.01,0.1,1,10,100],
        'model__class_weight':[None,'balanced']
    }

    # 4.创建GridSearchCV
    grid_search_cv = GridSearchCV(estimator=pipeline,
                            param_grid=params_range,# 使用宏平均F1选择最佳参数
                            scoring={
                                "f1_macro": "f1_macro",
                                "accuracy": "accuracy",
                            },
                            #告诉grid_search_cv记录哪些指标，验证集的f1和准确率
                            cv=predefined_split,n_jobs=-1,
                            # 选择最佳参数后,不使用训练集 + 验证集重新训练最终模型
                            refit="f1_macro",)
                            #告诉grid_search_cv使用哪个指标来选择最佳参数
    
    # 5.开始寻找参数
    grid_search_cv.fit(X=search_texts,y=search_labels)

    # 6.得到最优参数
    best_c = grid_search_cv.best_params_["model__C"]
    best_class_weight = grid_search_cv.best_params_["model__class_weight"]
    best_f1_score = grid_search_cv.best_score_
    best_index = grid_search_cv.best_index_
    best_valid_accuracy = grid_search_cv.cv_results_["mean_test_accuracy"][best_index]

    # 7.取模型和TF-IDF
    best_tfidf = grid_search_cv.best_estimator_.named_steps["tfidf"]
    best_model = grid_search_cv.best_estimator_.named_steps["model"]

    return {
        'tfidf':best_tfidf,
        'tfidf_conf':{
            'analyzer':'char',
            'ngram_range':(1,2)
        },
        'model':best_model,        
        'model_conf':{
            'c-value':best_c,
            'class_weight':best_class_weight,
            'max_iter':max_iter,
            'solver':"lbfgs",
            'random_state':random_state,
        },
        'metrics':
        {
            'best_valid_f1_score':best_f1_score,
            'best_valid_accuracy':best_valid_accuracy

        },
    }


def save_results(results):

    # 1. 保存模型和TF-IDF
    import joblib
    model_path = outputs_path / 'model.joblib'
    model = results['model']
    joblib.dump(model, model_path)

    tfidf_path = outputs_path / "tfidf.joblib"
    joblib.dump(results["tfidf"], tfidf_path)


    # 2.保存模型参数
    import json
    model_conf_path = outputs_path / 'model_conf.json'
    model_conf = results['model_conf']
    with model_conf_path.open('w',encoding='utf-8') as model_file:   
         json.dump(
            model_conf,
            model_file,
            ensure_ascii=False, # 写中文的格式
            indent=2,# 缩进2格
         )

    # 3.保存验证集指标
    valid_metrics_path = outputs_path / 'valid_metrics.json'
    metrics = results['metrics']
    with valid_metrics_path.open('w',encoding='utf-8') as valid_metrics_file:   
         json.dump(
            metrics,
            valid_metrics_file,
            ensure_ascii=False, # 写中文的格式
            indent=2,# 缩进2格
         )

    # 4.保存tfidf:
    tfidf_conf_path = outputs_path / 'tfidf_conf.json'
    tfidf = results['tfidf_conf']
    with tfidf_conf_path.open('w',encoding='utf-8') as tfidf_file:   
         json.dump(
            tfidf,
            tfidf_file,
            ensure_ascii=False, # 写中文的格式
            indent=2,# 缩进2格
         )


def main():
    data = prepare_data(train_path,valid_path,test_path)
    '''{
        "train_texts": train_texts,
        "train_labels": train_labels,
        "valid_texts": valid_texts,
        "valid_labels": valid_labels,
        "test_texts":test_texts,
        "test_labels":test_labels
    }'''

    random_state = 15
    max_iter = 1000
    results = run_experiment(data,max_iter,random_state) #内部会转成dataset
    '''{
        'tfidf':dataset['tfidf'],
        'model':model,
        'metrics':metrics,
        'model_conf':{
            'c_value':c_value,
            'max_iter':max_iter,
            'random_state':random_state,
        },
        'tfidf_conf':{
            'analyzer':'char',
            'ngram_range':(1,2)
        }
    }'''

    save_results(results)


if __name__ == "__main__":
    main()
    print('train done')