# python -m Param_Comparison.param_effect_regularization

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from models.lenet import LeNet
import pandas as pd
import os
from utils import (
    compute_metrics,
    print_metrics_inline
)

# 正则化组合设置
regularization_settings = [
    {"name": "None", "l2": 0.0, "dropout": 0.0},
    {"name": "L2", "l2": 1e-4, "dropout": 0.0},
    {"name": "Dropout", "l2": 0.0, "dropout": 0.5},
    {"name": "L2+Dropout", "l2": 1e-4, "dropout": 0.5}
]

def train_and_eval(setting):
    print(f"\n🚀 实验：正则化 = {setting['name']}")
    
    batch_size = 64
    lr = 0.001
    weight_decay = setting['l2']
    dropout_rate = setting['dropout']

    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LeNet(dropout_rate=dropout_rate).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss()

    log_rows = []

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
        acc_train, prec_train, rec_train, f1_train = compute_metrics(y_true, y_pred)
        print_metrics_inline(f"[Train Epoch {epoch+1}]", avg_train_loss, acc_train, prec_train, rec_train, f1_train)

        # 测试
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
        acc_test, prec_test, rec_test, f1_test = compute_metrics(y_true, y_pred)
        print_metrics_inline(f"[Test  Epoch {epoch+1}]", avg_test_loss, acc_test, prec_test, rec_test, f1_test)

        # 记录每轮日志
        log_rows.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'train_acc': acc_train,
            'train_precision': prec_train,
            'train_recall': rec_train,
            'train_f1': f1_train,
            'test_loss': avg_test_loss,
            'test_acc': acc_test,
            'test_precision': prec_test,
            'test_recall': rec_test,
            'test_f1': f1_test
        })

    # 保存日志文件
    log_dir = "Param_Comparison/regularization_logs"
    os.makedirs(log_dir, exist_ok=True)
    filename = f"{setting['name'].replace('+', '_')}_log.csv"
    save_path = os.path.join(log_dir, filename)
    pd.DataFrame(log_rows).to_csv(save_path, index=False)
    print(f"📄 日志保存至: {save_path}")

    return {
        "regularization": setting["name"],
        "accuracy": acc_test,
        "precision": prec_test,
        "recall": rec_test,
        "f1": f1_test
    }

def run_experiments():
    results = []
    for setting in regularization_settings:
        result = train_and_eval(setting)
        results.append(result)

    df = pd.DataFrame(results)
    save_path = "Param_Comparison/param_effect_regularization_results.csv"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"\n实验结果已保存到：{save_path}")

if __name__ == "__main__":
    run_experiments()
