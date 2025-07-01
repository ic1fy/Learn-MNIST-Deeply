# python -m TransferLearning.fine_tuning --model lenet --weights checkpoints/lenet_lr0.001_bs64.pt

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import copy
import os

from models.lenet import LeNet
from models.mobilenet import MobileNet
from models.resnet import ResNet
from models.vgg import VGG
from models.densenet import DenseNet
from models.vit import VisionTransformer
from utils import compute_metrics, print_metrics_inline
from TransferLearning.dataset_hand.custom_mnist import CustomMNISTDataset

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True, choices=['lenet', 'mobilenet', 'resnet', 'vgg', 'densenet', 'vit'])
parser.add_argument('--weights', type=str, required=True, help='模型权重路径（.pt）')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--no_progress', action='store_true')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_map = {
    'lenet': LeNet,
    'mobilenet': MobileNet,
    'resnet': ResNet,
    'vgg': VGG,
    'densenet': DenseNet,
    'vit': VisionTransformer
}
model_class = model_map[args.model]
model = model_class(num_classes=10, input_channels=1).to(device)
model.load_state_dict(torch.load(args.weights, map_location=device))
print(f"模型已加载: {args.weights}")

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = CustomMNISTDataset(
    image_path="TransferLearning/dataset_hand/train-images-idx3-ubyte",
    label_path="TransferLearning/dataset_hand/train-labels-idx1-ubyte",
    transform=transform
)
test_dataset = CustomMNISTDataset(
    image_path="TransferLearning/dataset_hand/t10k-images-idx3-ubyte",
    label_path="TransferLearning/dataset_hand/t10k-labels-idx1-ubyte",
    transform=transform
)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)

def evaluate(title="评估"):
    model.eval()
    y_true, y_pred = [], []
    total_loss = 0

    with torch.no_grad():
        loader = tqdm(test_loader, desc=title) if not args.no_progress else test_loader
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            preds = output.argmax(dim=1)
            y_true.extend(target.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            total_loss += loss.item()

    avg_loss = total_loss / len(test_loader)
    acc, precision, recall, f1 = compute_metrics(y_true, y_pred)
    print_metrics_inline(f"[{title}]", avg_loss, acc, precision, recall, f1)
    return avg_loss

def finetune():
    model.train()
    best_loss = float("inf")
    best_model_state = None
    patience = 3
    patience_counter = 0

    for epoch in range(1, 1000): 
        y_true, y_pred = [], []
        total_loss = 0

        loader = tqdm(train_loader, desc=f"Finetune Epoch {epoch}") if not args.no_progress else train_loader
        for data, target in loader:
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            preds = output.argmax(dim=1)
            y_true.extend(target.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        acc, precision, recall, f1 = compute_metrics(y_true, y_pred)
        print_metrics_inline(f"[Finetune Epoch {epoch}]", avg_train_loss, acc, precision, recall, f1)

        avg_test_loss = evaluate(f"Test after Epoch {epoch}")

        if avg_test_loss < best_loss:
            best_loss = avg_test_loss
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}.")
                break

        model.train()

    if best_model_state:
        model.load_state_dict(best_model_state)
        original_name = os.path.splitext(os.path.basename(args.weights))[0]
        save_path = f"checkpoints/{original_name}_finetuned_best.pt"

        torch.save(best_model_state, save_path)
        print(f"最优模型保存至: {save_path}")

evaluate("原始模型评估")
finetune()
evaluate("微调后评估")
