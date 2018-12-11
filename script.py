import numpy as np
import os
from PIL import Image

import time

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import Adam

from torch.utils.data.sampler import SubsetRandomSampler
from torch.autograd import Variable

train_csv = os.path.join("data", "sign_mnist_train.csv")
train_archive = os.path.join("data", "sign_mnist_train.npy")

test_csv = os.path.join("data", "sign_mnist_test.csv")
test_archive = os.path.join("data", "sign_mnist_test.npy")
'''
if not os.path.exists(train_archive):
    print("Loading data from " + train_csv + " ...")
    training_set = np.genfromtxt(train_csv, dtype="uint8", skip_header=1, delimiter=",")
    print("Saving data to " + train_archive + " ...")
    np.save(train_archive, training_set)
else:
    print("Loading data from " + train_archive + " ...")
    training_set = np.load(train_archive)

if not os.path.exists(test_archive):
    print("Loading data from " + test_csv + " ...")
    test_set = np.genfromtxt(test_csv, dtype="uint8", skip_header=1, delimiter=",")
    print("Saving data to " + test_archive + " ...")
    np.save(test_archive, test_set)
else:
    print("Loading data from " + test_archive + " ...")
    test_set = np.load(test_archive)
print("Loading data: Done")
'''

#training_labels = training_set[:, 0]
#training_data = training_set[:, 1:]
#training_data = training_data.reshape((-1, 28, 28))
#test_labels = test_set[:,0]
#test_data = test_set[:,1:]
#test_data = test_data.reshape((-1,28,28))

#Image.fromarray(training_data[0]).save("test.png")

#print("done")


class SignMNISTDataset(Dataset):
    """Sign language MNIST dataset."""

    def __init__(self, csv_file, npy_file, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file.
            npy_file (string): Path to the npy file.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        if not os.path.isfile(npy_file):
            #print("Loading data from " + csv_file + " ...")
            data = np.genfromtxt(csv_file, dtype="uint8", skip_header=1, delimiter=",")
            #print("Saving data to " + npy_file + " ...")
            np.save(npy_file, data)
        else:
            #print("Loading data from " + npy_file + " ...")
            data = np.load(npy_file)
        #print("Data:")
        #print(data)
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

class SimpleCNN(torch.nn.Module):

    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=18, kernel_size=3, stride=1, padding=1) # out: [18 x 28 x 28]
        self.relu1 = torch.nn.ReLU()
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0) # out: [18 x 14 x 14]
        self.net = torch.nn.Sequential(self.conv1, self.relu1, self.pool1)
        
        self.fc1 = torch.nn.Linear(in_features=18 * 14 * 14, out_features=64) # out: [64 x 1]
        self.relu2 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(in_features=64, out_features=25) # out: [25 x 1]

    def forward(self, input):
        output = self.net(input)
        output = output.view(-1, 18 * 14 * 14)
        output = self.fc1(output)
        output = self.relu2(output)
        output = self.fc2(output)
        return output

train_transformations = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5))
])
train_set = SignMNISTDataset(train_csv, train_archive)
train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=4)

test_transformations = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5))
])
test_set = SignMNISTDataset(test_csv, test_archive)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False, num_workers=4)

'''
for i in range(len(test_set)):
    sample = test_set[i]
    print(i, sample['label'].shape, sample['image'].shape)
'''

'''
n_training_samples = training_set.shape[0]
train_sampler = SubsetRandomSampler(np.arange(n_training_samples, dtype=np.uint8))

n_val_samples = 5000
val_sampler = SubsetRandomSampler(np.arange(n_training_samples, n_training_samples + n_val_samples, dtype=np.uint8))

n_test_samples = 5000
test_sampler = SubsetRandomSampler(np.arange(n_test_samples, dtype=np.uint8))

def get_train_loader(batch_size):
    train_loader = torch.utils.data.DataLoader(training_data, batch_size=batch_size,
                                        sampler=train_sampler, num_workers=2)
    return(train_loader)

test_loader = torch.utils.data.DataLoader(training_data, batch_size=4, sampler=test_sampler, num_workers=2)
val_loader = torch.utils.data.DataLoader(test_data, batch_size=128, sampler=val_sampler, num_workers=2)
'''

def trainModel(model, batch_size, n_epochs, learning_rate):
    print("===== HYPERPARAMETERS =====")
    print("batch_size=", batch_size)
    print("epochs=", n_epochs)
    print("learning_rate=", learning_rate)
    print("=" * 30)

    #train_loader = get_train_loader(batch_size)
    n_batches = len(train_loader)
    
    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    loss_fn = torch.nn.CrossEntropyLoss()
    
    training_start_time = time.time()

    for epoch in range(n_epochs):

        running_loss = 0.0
        print_every = n_batches // 10
        start_time = time.time()
        total_train_loss = 0

        for i, data in enumerate(train_loader, 0):
            labels = data['label']
            inputs = data['image']
            inputs = inputs.type(torch.FloatTensor)
            #print(data)
            #print(labels)
            #print(inputs)

            #inputs, labels = Variable(inputs), Variable(labels)

            optimizer.zero_grad()

            outputs = model(inputs)
            #print(outputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            total_train_loss += loss.item()

            if (i + 1) % (print_every + 1) == 0:
                print("Epoch {}, {:d}% \t train_loss: {:.2f} took: {:.2f}s".format(
                    epoch + 1, int(100 * (i + 1) / n_batches), running_loss / print_every, time.time() - start_time))
                running_loss = 0.0
                start_time = time.time()

            total_val_loss = 0
            '''
            for inputs, labels in val_loader:
                inputs, labels = Variable(inputs), Variable(labels)

                val_outputs = model(inputs)
                val_loss_size = loss(val_outputs, labels)
                total_val_loss += val_loss_size.data[0]

            print("Validation loss = {:.2f}".format(total_val_loss / len(val_loader)))
            '''
        print("Training finished, took {:.2f}s".format(time.time() - training_start_time))

def main():
    print("main")
    model = SimpleCNN()
    trainModel(model, batch_size=32, n_epochs=5, learning_rate=0.001)

if __name__ == "__main__":
    main()