import os
import numpy as np
import PIL
from torch.utils.data import Dataset


class SignMNISTDataset(Dataset):
    """Sign language MNIST dataset."""

    def __init__(self, options, input_path, transform=None):
        """
        Args:
            input_path (string): Path to the csv file.
            transform (callable, optional): Optional transform to be applied on a sample.
        """

        csv_file = input_path
        npy_file = os.path.splitext(input_path)[0] + '.npy'

        if not os.path.isfile(npy_file):
            print("Loading data from " + options.root_relpath(csv_file) + " ...")
            data = np.genfromtxt(csv_file, dtype="uint8", skip_header=1, delimiter=",")
            print("Saving data to " + options.root_relpath(npy_file) + " ...")
            np.save(npy_file, data)
        else:
            print("Loading data from " + options.root_relpath(npy_file) + " ...")
            data = np.load(npy_file)

        self.labels = data[:, 0].astype(dtype="uint8")
        #self.images = (1.0 * data[:, 1:] / 255).reshape((-1, 1, 28, 28))
        self.images = data[:, 1:].astype(dtype="uint8").reshape((-1, 1, 28, 28))
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        image = self.images[idx]
        if self.transform:
            # Do some conversions for the transforms to work:
            imgn = image.reshape((28, 28)) # reshape for PIL
            imgp = PIL.Image.fromarray(imgn) # numpy to PIL
            imgp2 = self.transform(imgp) # apply transform
            imgn2 = np.array(imgp2) # PIL to numpy
            image = imgn2.reshape((1, 28, 28)) # reshape back for net
        image = (1.0 * image  / 255) # to floating point
        sample = {'label': label, 'image': image}
        return sample
