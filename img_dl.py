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

    def __init__(self, dir_path, transform=None, target_transform=None, dims=(128, 128)): #extension

        self.path = dir_path
        self.transform = transform
        self.target_transform = target_transform
        self.dims = dims
        #self.extension = extension

        subdirs = next(os.walk(self.path))[1]

        L = []
        for i, class_ in enumerate(subdirs):
            imgs = next(os.walk(os.path.join(self.path, class_)))[2]

            for img in imgs:
                #if self.extension in img: # Generalizar para varios tipos de archivo
                pathimg = os.path.join(self.path, class_, img)
                L.append([pathimg, i])
            # crear diccionario aqui

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

transformations = transforms.Compose([transforms.ToTensor()])#transforms.CenterCrop(50), 
animales = ImageDataset(path, transform=transformations)

animales_ds = DataLoader(animales, batch_size=4, shuffle=True)
print(len(animales_ds))
print(len(animales))
images, labels = next(iter(animales_ds))

print(f"Feature batch shape: {images.size()}")
print(f"Labels batch shape: {labels.size()}")
# img = images[0].squeeze()
label = labels[0]
# plt.imshow(img.permute(1, 2, 0), cmap="gray")
# plt.show()
print(f"Label: {label}")

fig = plt.figure(figsize=(16, 16))
fig.subplots_adjust(hspace=1 ,wspace=1)

ax1 = fig.add_subplot(2, 2, 1)
#ax1.title.set_text('imagen')
ax1.axis("off")
ax1.imshow(images[0].squeeze().permute(1, 2, 0), cmap="gray")

ax2 = fig.add_subplot(2, 2, 2)
#ax2.title.set_text('mask')
ax2.axis("off")
ax2.imshow(images[1].squeeze().permute(1, 2, 0), cmap="gray")

ax3 = fig.add_subplot(2, 2, 3)
#ax3.title.set_text('prediccion')
ax3.axis("off")
ax3.imshow(images[2].squeeze().permute(1, 2, 0), cmap="gray")

ax4 = fig.add_subplot(2, 2, 4)
#ax3.title.set_text('prediccion')
ax4.axis("off")
ax4.imshow(images[3].squeeze().permute(1, 2, 0), cmap="gray")

plt.show()

# interfaz: escoger carpete, extensiones, tama;o imagen