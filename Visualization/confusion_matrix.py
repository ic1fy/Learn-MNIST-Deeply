# python Visualization/confusion_matrix.py --model lenet --checkpoint_path checkpoints/lenet_lr0.001_bs64.pt

import os
import argparse
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.lenet import LeNet
from models.mobilenet import MobileNet
from models.resnet import ResNet
from models.vgg import VGG
from models.densenet import DenseNet
from models.vit import VisionTransformer

from utils import plot_confusion_matrix 

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True, help='选择模型: lenet, mobilenet, resnet, vgg, densenet, vit')
parser.add_argument('--checkpoint_path', type=str, required=True, help='模型参数文件路径')
parser.add_argument('--batch_size', type=int, default=1000)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_dict = {
    'lenet': LeNet,
    'mobilenet': MobileNet,
    'resnet': ResNet,
    'vgg': VGG,
    'densenet': DenseNet,
    'vit': VisionTransformer
}

model = model_dict[args.model](num_classes=10, input_channels=1).to(device)
checkpoint_path = args.checkpoint_path
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

# 推理
y_true, y_pred = [], []

with torch.no_grad():
    for data, target in tqdm(test_loader, desc="Running Inference"):
        data, target = data.to(device), target.to(device)
        output = model(data)
        preds = output.argmax(dim=1)
        y_true.extend(target.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())


save_path = os.path.join("Visualization", f"confusion_matrix_{args.checkpoint_path}.png")
plot_confusion_matrix(
    y_true=y_true,
    y_pred=y_pred,
    class_names=[str(i) for i in range(10)],
    normalize=True,
    title=f"{args.checkpoint_path} Confusion Matrix",
    save_path=save_path
)
