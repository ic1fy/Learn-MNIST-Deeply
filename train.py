import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import os

from models.lenet import LeNet
from models.mobilenet import MobileNet
from models.resnet import ResNet
from models.vgg import VGG
from models.densenet import DenseNet 
from models.vit import VisionTransformer
from utils import compute_metrics, print_metrics_inline

parser = argparse.ArgumentParser(description="Train MNIST with Different Models")

parser.add_argument('--model', type=str, default='vit', choices=['lenet', 'mobilenet', 'resnet', 'vgg', 'densenet','vit'], help='模型名称')
parser.add_argument('--lr', type=float, default=0.001, help='学习率')
parser.add_argument('--batch_size', type=int, default=64, help='训练批大小')
parser.add_argument('--epochs', type=int, default=20, help='训练轮数')
parser.add_argument('--no_progress', action='store_true', help='是否关闭 tqdm 进度条') # 默认为False
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_dict = {
    'lenet': LeNet,
    'mobilenet': MobileNet,
    'resnet': ResNet,
    'vgg': VGG,
    'densenet': DenseNet,
    'vit':VisionTransformer
}

model_class = model_dict[args.model]
model = model_class(num_classes=10, input_channels=1).to(device)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)

def train(epoch):
    model.train()
    y_true, y_pred = [], []
    total_loss = 0.0

    iterator = tqdm(train_loader, desc=f"[Train Epoch {epoch}]") if not args.no_progress else train_loader
    for data, target in iterator:
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = output.argmax(dim=1)
        y_true.extend(target.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

        if not args.no_progress:
            iterator.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(train_loader)
    acc, precision, recall, f1 = compute_metrics(y_true, y_pred)
    print_metrics_inline(f"[Train Epoch {epoch}]", avg_loss, acc, precision, recall, f1)

def test(epoch):
    model.eval()
    y_true, y_pred = [], []
    total_loss = 0.0

    with torch.no_grad():
        iterator = tqdm(test_loader, desc=f"[Test Epoch {epoch}]") if not args.no_progress else test_loader
        for data, target in iterator:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()

            preds = output.argmax(dim=1)
            y_true.extend(target.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    avg_loss = total_loss / len(test_loader)
    acc, precision, recall, f1 = compute_metrics(y_true, y_pred)
    print_metrics_inline(f"[Test Epoch {epoch}]", avg_loss, acc, precision, recall, f1)

    return avg_loss

best_loss = float('inf')
patience = 3
min_delta = 1e-4
patience_counter = 0

best_model_state = None

for epoch in range(1, args.epochs + 1):
    train(epoch)
    val_loss = test(epoch)

    if best_loss - val_loss > min_delta:
        best_loss = val_loss
        patience_counter = 0
        best_model_state = model.state_dict()  #
    else:
        patience_counter += 1

        if patience_counter >= patience:
            print(f"\n Early Stopping, 最优验证损失: {best_loss:.4f}")
            break

if best_model_state is not None:
    model.load_state_dict(best_model_state)

save_dir = "checkpoints"
os.makedirs(save_dir, exist_ok=True)
model_name = f"{args.model}_lr{args.lr}_bs{args.batch_size}.pt"
model_path = os.path.join(save_dir, model_name)
model.save(model_path)

print(f"\n最佳模型已保存至: {model_path}")

