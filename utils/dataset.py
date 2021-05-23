import glob
import numpy as np
import os
from PIL import Image, ImageOps
import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class DrosophilaDataset(Dataset):
    def __init__(self, image_dir, label_dir, image_transforms=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_transforms = image_transforms

        self.image_list = glob.glob(os.path.join(image_dir, '*.png'))
        self.label_list = glob.glob(os.path.join(label_dir, '*.png'))

        self.image_list.sort()
        self.label_list.sort()

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image = Image.open(self.image_list[idx])
        label = Image.open(self.label_list[idx])

        seed = np.random.randint(2147483647)
        if self.image_transforms is not None:
            random.seed(seed)
            image = self.image_transforms(image)
            random.seed(seed)
            label = self.image_transforms(label)

        return image, label

def main():
    image_dir = os.path.join('D:/pytorch/Drosophila_Brain/data/train', 'image')
    label_dir = os.path.join('D:/pytorch/Drosophila_Brain/data/train', 'label')
    image_transformers = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor()
    ])
    dataset = DrosophilaDataset(image_dir, label_dir, image_transformers)
    print(dataset.__len__())

if __name__ == '__main__':
    main()