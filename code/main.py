import os

import train
from datasets import SignMNISTDataset
from nets import SimpleCNN


def main():
    train_path = os.path.join("..", "data", "sign_mnist_train.csv")
    test_path = os.path.join("..", "data", "sign_mnist_test.csv")

    train_set = SignMNISTDataset.SignMNISTDataset(train_path)
    test_set  = SignMNISTDataset.SignMNISTDataset(test_path)

    model = SimpleCNN.SimpleCNN()
    train.trainModel(model, test_set, train_set, batch_size=32, n_epochs=5, learning_rate=0.001)


if __name__ == "__main__":
    main()
