from numpy import genfromtxt
import numpy as np
import os
from PIL import Image

import time

import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
from torch.autograd import Variable
import torch.nn.functional as F

train_archive = os.path.join("data", "sign_mnist_train.npy")
test_archive = os.path.join("data", "sign_mnist_test.npy")

if not os.path.exists(train_archive) or not os.path.exists(test_archive):
    training_set = genfromtxt(os.path.join("data", "sign_mnist_train.csv"), dtype="uint8", skip_header=1, delimiter=",")
    test_set = genfromtxt(os.path.join("data", "sign_mnist_test.csv"), dtype="uint8", skip_header=1, delimiter=",")
    np.save(train_archive, training_set)
    np.save(test_archive, test_set)
else:
    training_set = np.load(train_archive)
    test_set = np.load(test_archive)

#training_labels = training_set[:, 0]
#training_data = training_set[:, 1:]
#training_data = training_data.reshape((-1, 28, 28))
#test_labels = test_set[:,0]
#test_data = test_set[:,1:]
#test_data = test_data.reshape((-1,28,28))

#Image.fromarray(training_data[0]).save("test.png")

#print("done")


class SimpleCNN(torch.nn.Module):

    def __init__(self):
        super(SimpleCNN, self).__init__()

        self.conv1 = torch.nn.Conv2d(1, 18, kernel_size=3, stride=1, padding=1)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = torch.nn.Linear(18 * 14 * 14, 64)
        self.fc2 = torch.nn.Linear(64, 24)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = x.view(-1, 18 * 14 *14)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

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


def createLossAndOptimizer(net, learning_rate=0.001):
    loss = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    return (loss, optimizer)

def trainNet(net, batch_size, n_epochs, learning_rate):
    print("===== HYPERPARAMETERS =====")
    print("batch_size=", batch_size)
    print("epochs=", n_epochs)
    print("learning_rate=", learning_rate)
    print("=" * 30)

    train_loader = get_train_loader(batch_size)
    n_batches = len(train_loader)
    loss, optimizer = createLossAndOptimizer(net, learning_rate)
    training_start_time = time.time()

    for epoch in range(n_epochs):

        running_loss = 0.0
        print_every = n_batches // 10
        start_time = time.time()
        total_train_loss = 0

        for i, data in enumerate(train_loader, 0):
            inputs, labels = data

            inputs, labels = Variable(inputs), Variable(labels)

            optimizer.zero_grad()

            outputs = net(inputs)
            loss_size = loss(outputs, labels)
            loss_size.backward()
            optimizer.step()

            running_loss += loss_size.data[0]
            total_train_loss += loss_size.data[0]

            if (i + 1) % (print_every + 1) == 0:
                print("Epoch {}, {:d}% \t train_loss: {:.2f} took: {:.2f}s".format(
                    epoch + 1, int(100 * (i + 1) / n_batches), running_loss / print_every, time.time() - start_time))
                running_loss = 0.0
                start_time = time.time()

            total_val_loss = 0
            for inputs, labels in val_loader:
                inputs, labels = Variable(inputs), Variable(labels)

                val_outputs = net(inputs)
                val_loss_size = loss(val_outputs, labels)
                total_val_loss += val_loss_size.data[0]

            print("Validation loss = {:.2f}".format(total_val_loss / len(val_loader)))
            print("Training finished, took {:.2f}s".format(time.time() - training_start_time))

def main():
    CNN = SimpleCNN()
    trainNet(CNN, batch_size=32, n_epochs=5, learning_rate=0.001)

if __name__ == "__main__":
    main()