import torch
import nibabel as nib
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

class UNET_Dataset(Dataset):

	def __init__(self, path_file):

		self.path = path_file

	def __len__(self):

		train_ids = next(os.walk(self.path))[1]
		self.dims = np.zeros(len(train_ids))

		for i in range(len(train_ids)):
			img_path = os.path.join(self.path, train_ids[i])
			image = nib.load(os.path.join(img_path, 'T1.nii'))
			image_data = image.get_fdata()
			self.dims[i] = image_data.shape[2]

		return int(np.sum(self.dims))

	def __getitem__(self, idx):

		c_sum = np.cumsum(self.dims)

		f = c_sum < idx
		ind_f = int(np.where(f==False)[0][0])

		if ind_f > 0:
			ind_img = int(idx % c_sum[ind_f-1] - 1)
		else:
			ind_img = int(idx - 1)

		train_ids = next(os.walk(self.path))[1]
		img_path = os.path.join(self.path, train_ids[ind_f])

		image = nib.load(os.path.join(img_path, 'T1.nii'))
		image_data = image.get_fdata()

		mask = nib.load(os.path.join(img_path, 'LabelsForTraining.nii')) 
		mask_data = mask.get_fdata()

		return (torch.from_numpy(image_data[:,:,ind_img]), torch.from_numpy(mask_data[:,:,ind_img]))

DATA_PATH_IMG = '/media/david/datos1/Coding/maestria/trabajo_de_grado/databases/MRBrainS13DataNii/TrainingData'

cabezas = UNET_Dataset(DATA_PATH_IMG)

cabezas_dataset = DataLoader(cabezas, batch_size=80, shuffle=True)
print(cabezas_dataset)

images, labels = next(iter(cabezas_dataset))

print(images.shape)
print(labels.shape)

for slice_ in range(images.shape[0]):

	fig = plt.figure(figsize=(16,16))
	fig.subplots_adjust(hspace=1 ,wspace=1)

	ax1 = fig.add_subplot(1,2,1)
	ax1.title.set_text('imagen')
	ax1.axis("off")
	ax1.imshow(images[slice_,:,:],cmap="gray")

	ax2 = fig.add_subplot(1,2,2)
	ax2.title.set_text('mask')
	ax2.axis("off")
	ax2.imshow(labels[slice_, :, :],cmap="gray")

	plt.show()
