import torch
import torch.nn as nn
from torchvision.models import densenet121
from models.base_model import BaseModel
import torch.nn.functional as F

# densenet121
class DenseNet(BaseModel):
    def __init__(self, num_classes=10, input_channels=1):
        super(DenseNet, self).__init__()

        self.backbone = densenet121(weights=None)

        if input_channels != 3:
            self.backbone.features.conv0 = nn.Conv2d(
                input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )

        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        if x.shape[-1] < 32 or x.shape[-2] < 32:
            x = F.interpolate(x, size=(64, 64), mode='bilinear', align_corners=False)
        return self.backbone(x)
