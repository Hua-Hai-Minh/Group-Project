import os
from PIL import Image
import torch
from torch.utils.data import Dataset

class CelebADataset(Dataset):
    def __init__(self, root_dir, annotation_file, transform=None):
        self.root_dir = root_dir
        self.annotation_file = annotation_file
        self.transform = transform

        self.image_paths = []
        self.labels = []

        with open(annotation_file, 'r') as f:
            for line in f:
                image_path, label = line.strip().split()
                self.image_paths.append(os.path.join(root_dir, image_path))
                self.labels.append(int(label))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label
