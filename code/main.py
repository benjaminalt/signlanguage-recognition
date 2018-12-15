import os
import torch

import train
import options
from datasets.SignMNISTDataset import SignMNISTDataset
from nets.DeepCNN import DeepCNN
from visualizer.Visualizer import Visualizer


def main():
    train_path = os.path.join("..", "data", "sign_mnist_train.csv")
    test_path = os.path.join("..", "data", "sign_mnist_test.csv")

    opts = options.Options()

    train_set = SignMNISTDataset(train_path, opts)
    test_set  = SignMNISTDataset(test_path, opts)

    model = DeepCNN(opts)
    model = model.cuda() if torch.cuda.is_available() else model

    vis = Visualizer()

    if opts.gridSearch:
        for config in opts.iter():
            print("Testing {}".format(config))
            vis.show(lambda: train.trainModel(model, train_set, test_set, opts), opts)
    else:
        vis.show(lambda : train.trainModel(model, train_set, test_set, opts), opts)


if __name__ == "__main__":
    main()
