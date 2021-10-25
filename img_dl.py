import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

path = 'data/animals'

class ImageDataset(Dataset):

    def __init__(self, dir_path, transform=None, target_transform=None):

        self.path = dir_path
        self.transform = transform
        self.target_transform = target_transform
        self.dims = (128, 128)

        subdirs = next(os.walk(self.path))[1]

        L = []
        for i, class_ in enumerate(subdirs):
            imgs = next(os.walk(os.path.join(self.path, class_)))[2]

            for img in imgs:
                pathimg = os.path.join(self.path, class_, img)
                L.append([pathimg, i])

        self.df = pd.DataFrame(L, columns=['Path', 'Class'])

    def __len__(self):

        return self.df.shape[0]

    def __getitem__(self, idx):

        img_path = self.df.at[idx, 'Path']
        label = self.df.at[idx, 'Class']

        image = Image.open(img_path)
        image = image.resize(self.dims)

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
            
        return image, label

transformations = transforms.Compose([transforms.CenterCrop(50), transforms.ToTensor()])#
animales = ImageDataset(path, transform=transformations)

animales_ds = DataLoader(animales, batch_size=2, shuffle=True)
print(animales_ds)

images, labels = next(iter(animales_ds))

print(f"Feature batch shape: {images.size()}")
print(f"Labels batch shape: {labels.size()}")
img = images[0].squeeze()
label = labels[0]
plt.imshow(img.permute(1, 2, 0), cmap="gray")
plt.show()
print(f"Label: {label}")