import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base_model import BaseModel

class LeNet(BaseModel):
    def __init__(self, num_classes=10, input_channels=1, activation=nn.Tanh(), dropout_rate=0.0):
        super(LeNet, self).__init__()
        self.activation = activation
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

        self.conv1 = nn.Conv2d(input_channels, 6, kernel_size=5)
        self.pool1 = nn.AvgPool2d(2, 2)

        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool2 = nn.AvgPool2d(2, 2)

        self.conv3 = nn.Conv2d(16, 120, kernel_size=5)

        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, num_classes)

    def forward(self, x):
        # MNIST 输入是 28x28，需要 pad 到 32x32
        if x.shape[-1] == 28 and x.shape[-2] == 28:
            x = F.pad(x, pad=(2, 2, 2, 2))

        x = self.pool1(self.activation(self.conv1(x)))
        x = self.pool2(self.activation(self.conv2(x)))
        x = self.activation(self.conv3(x))
        x = x.view(-1, 120)
        x = self.dropout(self.activation(self.fc1(x)))
        x = self.fc2(x)
        return x
