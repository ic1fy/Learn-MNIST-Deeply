from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch.nn as nn
import torch.optim as optim
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os

def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    return acc, precision, recall, f1

def print_metrics_inline(title, avg_loss, acc, precision, recall, f1):
    print(f"{title} Loss: {avg_loss:.4f} | Acc: {acc:.4f} | Prec: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
    

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss()

    def forward(self, inputs, targets):
        logpt = -self.ce(inputs, targets)
        pt = torch.exp(logpt)
        loss = ((1 - pt) ** self.gamma) * -logpt
        return loss.mean()

def get_activation(name):
    activations = {
        'ReLU': nn.ReLU(),
        'Tanh': nn.Tanh()
    }
    return activations[name]

def get_optimizer(name, params, lr):
    if name == 'SGD':
        return optim.SGD(params, lr=lr)
    elif name == 'Adam':
        return optim.Adam(params, lr=lr)
    else:
        raise ValueError(f"Unsupported optimizer: {name}")

def get_loss_fn(name):
    if name == 'CrossEntropy':
        return nn.CrossEntropyLoss()
    elif name == 'Focal':
        return FocalLoss()
    else:
        raise ValueError(f"Unsupported loss function: {name}")


def plot_confusion_matrix(y_true, y_pred, class_names=None, normalize=False, title="Confusion Matrix", save_path=None):
    """
    在 utils.py 中定义绘制混淆矩阵的函数
    """
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(include_values=True, cmap=plt.cm.Blues, xticks_rotation='horizontal')
    plt.title(title)
    
    # 设置矩阵中数字的字体大小
    for text in disp.text_.ravel():
        text.set_fontsize(7)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"[INFO] 混淆矩阵已保存至: {save_path}")
    else:
        plt.show()
