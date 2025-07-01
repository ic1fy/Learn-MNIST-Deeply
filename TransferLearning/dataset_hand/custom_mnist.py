import torch
from torch.utils.data import Dataset
import numpy as np

class CustomMNISTDataset(Dataset):
    def __init__(self, image_path, label_path, transform=None):
        self.transform = transform
        self.images = self._read_images(image_path)
        self.labels = self._read_labels(label_path)

    def _read_images(self, path):
        with open(path, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
            return data.reshape(-1, 28, 28)

    def _read_labels(self, path):
        with open(path, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
            return data

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = int(self.labels[idx])
        image = torch.tensor(image, dtype=torch.uint8)

        if self.transform:
            image = self.transform(image)

        return image, label
