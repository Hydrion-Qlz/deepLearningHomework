import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        # Convert arrays to tensors
        image = torch.tensor(self.images[index], dtype=torch.float32)
        label = torch.tensor(self.labels[index], dtype=torch.float32)
        return image, label
