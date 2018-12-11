import os
import numpy as np
from torch.utils.data import Dataset


class SignMNISTDataset(Dataset):
    """Sign language MNIST dataset."""

    def __init__(self, input_path, transform=None):
        """
        Args:
            input_path (string): Path to the csv file.
            transform (callable, optional): Optional transform to be applied on a sample.
        """

        csv_file = input_path
        npy_file = os.path.splitext(input_path)[0]+'.npy'

        if not os.path.isfile(npy_file):
            print("Loading data from " + csv_file + " ...")
            data = np.genfromtxt(csv_file, dtype="uint8", skip_header=1, delimiter=",")
            print("Saving data to " + npy_file + " ...")
            np.save(npy_file, data)
        else:
            print("Loading data from " + npy_file + " ...")
            data = np.load(npy_file)

        self.labels = data[:, 0].astype(dtype="int64")
        self.images = (1.0 * data[:, 1:] / 255).reshape((-1, 1, 28, 28))
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        image = self.images[idx]
        if self.transform:
            image = self.transform(image)
        sample = {'label': label, 'image': image}
        return sample
