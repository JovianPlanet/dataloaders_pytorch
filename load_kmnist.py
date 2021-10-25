import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

# datos de entrenamiento
training_data = datasets.KMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

# Datos de prueba
test_data = datasets.KMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

# Etiquetas
labels_map = {
    0: "O",
    1: "Ki",
    2: "Su",
    3: "Tsu",
    4: "Na",
    5: "Ha",
    6: "Ma",
    7: "Ya",
    8: "Re",
    9: "Wo",
}

# plot de los datos
figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()