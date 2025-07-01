# python -m Param_Comparison.param_effect_analysis

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from models.lenet import LeNet
import pandas as pd
import os
from utils import (
    compute_metrics,
    print_metrics_inline,
    get_activation,
    get_optimizer,
    get_loss_fn
)

# 超参数组合
param_grid = {
    'batch_size': [32, 64],
    'learning_rate': [0.01, 0.001],
    'optimizer': ['SGD', 'Adam'],
    'activation': ['ReLU', 'Tanh'],
    'loss_fn': ['CrossEntropy', 'Focal']
}

def train_and_eval(batch_size, lr, optimizer_name, activation_name, loss_name):
    print(f"\n🚀 实验: BS={batch_size}, LR={lr}, OPT={optimizer_name}, ACT={activation_name}, LOSS={loss_name}")

    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LeNet(activation=get_activation(activation_name)).to(device)
    optimizer = get_optimizer(optimizer_name, model.parameters(), lr)
    criterion = get_loss_fn(loss_name)

    for epoch in range(10):
        model.train()
        total_loss = 0
        y_true, y_pred = [], []

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = outputs.argmax(dim=1).detach().cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(labels.detach().cpu().numpy())

        avg_train_loss = total_loss / len(train_loader)
        acc, precision, recall, f1 = compute_metrics(y_true, y_pred)
        print_metrics_inline(f"[Train Epoch {epoch+1}]", avg_train_loss, acc, precision, recall, f1)

        # 测试阶段
        model.eval()
        test_loss = 0
        y_true, y_pred = [], []
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()

                preds = outputs.argmax(dim=1).detach().cpu().numpy()
                y_pred.extend(preds)
                y_true.extend(labels.detach().cpu().numpy())

        avg_test_loss = test_loss / len(test_loader)
        acc, precision, recall, f1 = compute_metrics(y_true, y_pred)
        print_metrics_inline(f"[Test  Epoch {epoch+1}]", avg_test_loss, acc, precision, recall, f1)

    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def run_experiments():
    results = []
    for bs in param_grid['batch_size']:
        for lr in param_grid['learning_rate']:
            for opt in param_grid['optimizer']:
                for act in param_grid['activation']:
                    for loss_fn in param_grid['loss_fn']:
                        metrics = train_and_eval(bs, lr, opt, act, loss_fn)
                        result = {
                            'batch_size': bs,
                            'learning_rate': lr,
                            'optimizer': opt,
                            'activation': act,
                            'loss_fn': loss_fn,
                            **metrics
                        }
                        results.append(result)

    df = pd.DataFrame(results)
    save_path = "Param_Comparison/param_experiments_results.csv"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"\n所有实验结果已保存至：{save_path}")

if __name__ == "__main__":
    run_experiments()
