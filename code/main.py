import os
import torch

import train
from datasets.SignMNISTDataset import SignMNISTDataset
from nets.DeepCNN import DeepCNN
from visualizer.Visualizer import Visualizer


def main():
    train_path = os.path.join("..", "data", "sign_mnist_train.csv")
    test_path = os.path.join("..", "data", "sign_mnist_test.csv")

    train_set = SignMNISTDataset(train_path)
    test_set  = SignMNISTDataset(test_path)

    model = DeepCNN()
    model = model.cuda() if torch.cuda.is_available() else model

    vis = Visualizer()
    vis.show(lambda : train.trainModel(model, train_set, test_set, batch_size=32, n_epochs=10, learning_rate=0.001))


if __name__ == "__main__":
    main()
