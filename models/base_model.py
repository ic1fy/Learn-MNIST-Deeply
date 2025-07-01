import torch
import torch.nn as nn
import os

class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x):
        raise NotImplementedError(
            "[ERROR] 子类必须实现 forward() 方法！"
        )

    def save(self, path):
        """
        保存模型参数到 path 文件
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)
        print(f"[INFO] 模型参数已保存到：{path}")

    def load(self, path, map_location=None):
        """
        从 path 加载模型参数
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"[ERROR] 找不到模型文件: {path}")
        self.load_state_dict(torch.load(path, map_location=map_location or self.device))
        self.to(self.device)
        print(f"[INFO] 模型已从 {path} 成功加载")

    def summary(self, input_size=(1, 28, 28)):
        """
        打印模型结构与参数统计
        """
        try:
            from torchsummary import summary
            self.to(self.device)
            summary(self, input_size)
        except ImportError:
            print("[WARN] 未安装 torchsummary，使用基础信息")
            print(self)
            total = sum(p.numel() for p in self.parameters())
            print(f"[INFO] 模型参数总量: {total}")

    def predict(self, x):
        """
        简化推理流程（batch 或单张图像）
        输入张量需已标准化为 [B, C, H, W]
        """
        self.eval()
        x = x.to(self.device)
        with torch.no_grad():
            output = self.forward(x)
            predicted = torch.argmax(output, dim=1)
        return predicted
