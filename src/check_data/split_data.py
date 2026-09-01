from pathlib import Path
import random

# 1.设置路径
project_path = Path(__file__).parent.parent.parent
data_path = project_path / 'data' / 'tickets_1000.csv'
csv_path = project_path / 'data'
outputs_path = project_path / 'outputs'
outputs_path.mkdir(parents=True,exist_ok=True)
check_data_path = outputs_path / "check_data"
train_csv_path = csv_path / 'train.csv'
valid_csv_path = csv_path / 'valid.csv'
test_csv_path = csv_path / 'test.csv'
train_distribution_csv_path = check_data_path / 'train_distribution.csv'
valid_distribution_csv_path = check_data_path / 'valid_distribution.csv'
test_distribution_csv_path = check_data_path / 'test_distribution.csv'

# 2.准备数据集
seed = 15
import pandas as pd
dataframe = pd.read_csv(data_path)

# 按场景模板分组切分，防止同一模板的不同改写同时出现在训练集和测试集。
# 当前每个类别有 20 个 template_id，按 14/3/3 分配后仍为 70%/15%/15%。
if "template_id" not in dataframe.columns:
    raise ValueError("数据缺少 template_id，无法进行无模板泄漏的分组切分")

rng = random.Random(seed)
split_template_ids = {"train": set(), "valid": set(), "test": set()}

for label, label_dataframe in dataframe.groupby("label", sort=True):
    template_ids = sorted(label_dataframe["template_id"].unique().tolist())
    if len(template_ids) < 3:
        raise ValueError(f"类别 {label} 的 template_id 少于 3 个，无法切分")

    rng.shuffle(template_ids)
    train_end = round(len(template_ids) * 0.70)
    valid_end = train_end + round(len(template_ids) * 0.15)
    split_template_ids["train"].update(template_ids[:train_end])
    split_template_ids["valid"].update(template_ids[train_end:valid_end])
    split_template_ids["test"].update(template_ids[valid_end:])

train_dataframe = dataframe[dataframe["template_id"].isin(split_template_ids["train"])].copy()
valid_dataframe = dataframe[dataframe["template_id"].isin(split_template_ids["valid"])].copy()
test_dataframe = dataframe[dataframe["template_id"].isin(split_template_ids["test"])].copy()

# 打乱行序只影响文件展示，不破坏 template_id 的分组归属。
train_dataframe = train_dataframe.sample(frac=1, random_state=seed)
valid_dataframe = valid_dataframe.sample(frac=1, random_state=seed)
test_dataframe = test_dataframe.sample(frac=1, random_state=seed)

'''重置索引'''
# train_dataframe = train_dataframe.reset_index(drop=True)
# valid_dataframe = valid_dataframe.reset_index(drop=True)
# test_dataframe = test_dataframe.reset_index(drop=True) 
# 因为随机划分会打乱数据的索引，所以重新创建csv时需要重新划分索引

# 3. 简单的结果统计
count_train = train_dataframe['label'].value_counts().rename_axis('label').reset_index(name='count')
count_train['per'] = (count_train['count'] / train_dataframe['label'].notna().sum() * 100).round(2)

count_valid = valid_dataframe['label'].value_counts().rename_axis('label').reset_index(name='count')
count_valid['per'] = (count_valid['count'] / valid_dataframe['label'].notna().sum() * 100).round(2)

count_test = test_dataframe['label'].value_counts().rename_axis('label').reset_index(name='count')
count_test['per'] = (count_test['count'] / test_dataframe['label'].notna().sum() * 100).round(2)

# 4. 保存结果
count_train.to_csv(train_distribution_csv_path,index=False,encoding='utf-8')
count_valid.to_csv(valid_distribution_csv_path,index=False,encoding='utf-8')
count_test.to_csv(test_distribution_csv_path,index=False,encoding='utf-8')

train_dataframe.to_csv(train_csv_path,index=False,encoding='utf-8')
valid_dataframe.to_csv(valid_csv_path,index=False,encoding='utf-8')
test_dataframe.to_csv(test_csv_path,index=False,encoding='utf-8')
