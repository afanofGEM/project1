import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
TRAIN_PATH =  DATA_ROOT /"train.csv"
TEST_PATH = DATA_ROOT /"test.csv"

TEXT_COLUMN = "text"
LABEL_COLUMN = "label"

CLASS_0 = "app"
CLASS_1 = "chat"

MAX_LEN = 32
BATCH_SIZE = 32
EMBEDDING_DIM = 64
EPOCHS = 50
LR = 1e-3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed()

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

'''对训练集和测试集进行瘦身
只保留text label列，以及类别为网络、费用的样本'''
train_df = train_df[train_df[LABEL_COLUMN].isin([CLASS_0, CLASS_1])].copy()
test_df = test_df[test_df[LABEL_COLUMN].isin([CLASS_0, CLASS_1])].copy()

label_to_id = {
    CLASS_0: 0,
    CLASS_1: 1,
}

train_df[LABEL_COLUMN] = train_df[LABEL_COLUMN].map(label_to_id)
test_df[LABEL_COLUMN] = test_df[LABEL_COLUMN].map(label_to_id)


# 4. 建立字符词表
char_to_id = {
    '<PAD>': 0,
    '<UNK>': 1,
}

for text in train_df[TEXT_COLUMN]:
    for char in text:
        if char not in char_to_id:
            char_to_id[char] = len(char_to_id)

print("\nVocab size:", len(char_to_id))


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


# 6. Dataset
class TicketDataset(Dataset):
    def __init__(self, texts, labels, char_to_id, max_len):
        self.texts = texts.tolist()
        self.labels = labels.tolist()
        self.char_to_id = char_to_id
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        input_ids = encode_text(
            str(self.texts[index]),
            self.char_to_id,
            self.max_len,
        )
        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(self.labels[index], dtype=torch.float32),
        )

train_dataset = TicketDataset(train_df[TEXT_COLUMN],train_df[LABEL_COLUMN],
                              char_to_id,MAX_LEN)
test_dataset = TicketDataset(test_df[TEXT_COLUMN],test_df[LABEL_COLUMN],
                              char_to_id,MAX_LEN)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)



# 7. 二分类 MLP
from ..src.mlp.mlp_model import TicketClassifierMLP



'''损失函数1：二元交叉熵'''
class BCELoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, output, target):
        return self.loss(output, target)


# 9. Hinge Loss
class HingeLoss(nn.Module):

    def forward(self, output, target):

        # 0 / 1
        # ↓
        # -1 / +1
        y = target * 2 - 1

        margin = y * output

        '''判断距离边界1的距离，大于1就不惩罚'''
        loss = torch.clamp(1 - margin, min=0) # 如果小于0，就变成0

        return loss.mean()


# 10. Exponential Loss
class ExponentialLoss(nn.Module):

    def forward(self, output, target):

        y = target * 2 - 1

        margin = y * output

        loss = torch.exp(-margin)

        return loss.mean()


# 11. EnsLoss
class EnsLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.lam = 0.0

    def bc_inverse(self, z):

        if self.lam == 0:
            return torch.exp(z)

        value = 1 + self.lam * z

        value = torch.clamp(value, min=0)

        return value ** (1 / self.lam)

    def forward(self, output, target):

        # 0 / 1 → -1 / +1
        y = target * 2 - 1

        # margin = y * f(x)
        '''margin 很大而且 > 0 → 预测正确，而且很稳
           margin 接近 0 → 在分类边界附近
           margin < 0 → 预测错了'''
        margin = output * y

        '''接下来要给样本分配不同的梯度'''
        batch_size = len(target)

        random_values = torch.randn(batch_size,device=output.device)

        '''约等于≈torch.exp(z)'''
        random_gradients = -self.bc_inverse(random_values)

        # 排序
        random_gradients, _ = torch.sort(random_gradients)
        random_gradients = torch.clamp(random_gradients,min=-1.0)
        _, indices = torch.sort(margin.detach())
        gradients = torch.zeros_like(margin)


        '''假设：
            margin 排序：

            -1.0    0.5    2.0
            最差    一般    很好
        而随机 gradient 排序：
            -1.0   -0.7   -0.2
        于是 EnsLoss 配成：
            margin = -1.0
            gradient = -1.0 预测很差：使劲改

            margin = 0.5
            gradient = -0.7 预测一般：改一些

            margin = 2.0
            gradient = -0.2 已经预测很好：轻轻改。'''
        gradients[indices] = random_gradients


        # margin 很大时，说明已经训练的很好了，再调整
        large_margin = margin.detach() > 1
        gradients[large_margin] *= (
            1 / margin.detach()[large_margin]
        )

        '''这份梯度表已经定好了，别追究它祖宗十八代'''
        gradients = gradients.detach() - 1e-6

        '''先人为生成：这个样本的梯度 = -0.7
            然后构造：loss = -0.7 * margin
            PyTorch：loss.backward()
            一求导：gradient = -0.7
            因此：
                loss = gradients * margin
            本质上是在：
                把随机生成的梯度塞进PyTorch的autograd(自动求导)系统。'''
        loss = gradients * margin

        return loss.mean()


# 12. 测试模型
def evaluate(model):

    model.eval()

    labels = []
    predictions = []

    with torch.no_grad():

        for input_ids, label in test_loader:

            input_ids = input_ids.to(DEVICE)

            logits = model(input_ids).squeeze(-1)

            probabilities = torch.sigmoid(logits)

            prediction = (probabilities >= 0.5).long()

            labels.extend(label.long().tolist())
            predictions.extend(prediction.cpu().tolist())

    accuracy = accuracy_score(labels, predictions)

    f1 = f1_score(labels,predictions,zero_division=0)

    return accuracy, f1



# 13. 训练一个 loss
def train_one_loss(loss_name, criterion):

    print("Loss:", loss_name)
    # 每一个 loss 都重新初始化完全相同的模型
    set_seed()

    model = TicketClassifierMLP(
        vocab_size=len(char_to_id),
        embedding_dim=EMBEDDING_DIM,
        padding_id=char_to_id['<PAD>'],
        hidden_dim=128,
        dropout=0.3,
        num_classes=1,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
    )

    best_accuracy = 0

    for epoch in range(1, EPOCHS + 1):

        model.train()

        total_loss = 0

        for input_ids, labels in train_loader:

            input_ids = input_ids.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            logits = model(input_ids).squeeze(-1)

            loss = criterion(
                logits,
                labels,
            )

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        accuracy, f1 = evaluate(model)

        best_accuracy = max(
            best_accuracy,
            accuracy,
        )

        if epoch == 1 or epoch % 5 == 0:

            print(
                f"Epoch {epoch:02d} | "
                f"Loss {total_loss / len(train_loader):.4f} | "
                f"Accuracy {accuracy:.4f} | "
                f"F1 {f1:.4f}"
            )

    final_accuracy, final_f1 = evaluate(model)

    return {
        "loss": loss_name,
        "final_accuracy": final_accuracy,
        "final_f1": final_f1,
        "best_accuracy": best_accuracy,
    }


# ============================================================
# 14. 正式比较四种 loss
# ============================================================

loss_functions = {
    "BCE": BCELoss(),
    "Hinge": HingeLoss(),
    "Exponential": ExponentialLoss(),
    "EnsLoss": EnsLoss(),
}

results = []

for loss_name, criterion in loss_functions.items():

    result = train_one_loss(
        loss_name,
        criterion,
    )

    results.append(result)


# ============================================================
# 15. 输出最终结果
# ============================================================

results_df = pd.DataFrame(results)

print("\n")
print("=" * 60)
print("Final Results")
print("=" * 60)

print(
    results_df.sort_values(
        by="best_accuracy",
        ascending=False,
    )
)
