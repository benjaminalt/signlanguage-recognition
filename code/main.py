import os
import torch
import hiddenlayer as hl
import csv
import argparse

from train import ModelTrainer
import options

# from datasets.SignMNISTDataset import SignMNISTDataset
from torchvision.datasets.mnist import MNIST
from torchvision import transforms

from nets import *
from visualizer.Visualizer import Visualizer

def main(args):
    opts = options.Options()

    # Create output directory:
    os.makedirs(opts.output_dir(), exist_ok=True)

    # train_path = opts.data_path("sign_mnist_train.csv")
    # test_path = opts.data_path("sign_mnist_test.csv")

    # train_set = SignMNISTDataset(opts, train_path)
    # test_set  = SignMNISTDataset(opts, test_path)


    transform = transforms.Compose([
        # you can add other transformations in this list
        transforms.ToTensor()
    ])

    train_set = MNIST(opts.data_dir(), train=True, download=True, transform=transform)
    test_set = MNIST(opts.data_dir(), train=False, download=True, transform=transform)

    model = CNNs.CNN_5(opts)
    
    # note: hiddenlayer library doesn't seem to work with the cuda variant of the model
    model_graph_file = opts.output_path("model")
    print("Generating model graph visualization at " + opts.root_relpath(model_graph_file) + " ...")
    hl.build_graph(model, torch.zeros([1,1,28,28])).save(model_graph_file)

    # Check if using cuda:
    if opts.use_cuda:
        print("Using CUDA!")
        model = model.cuda()
    else:
        print("Not using CUDA!")

    trainer = ModelTrainer()
    vis = Visualizer(opts)

    # Train, visualize & document results in CSV file
    with open(opts.output_path("results.csv"), "w") as csv_file:
        fieldnames = opts.var_names()
        fieldnames.extend(["final_test_loss", "final_test_acc"])
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        if opts.gridSearch:
            for config in opts.iter():
                print("Testing {}".format(config))
                vis.show(lambda: trainer.trainModel(model, train_set, test_set, opts))
                csv_dict = opts.values()
                csv_dict.update({"final_test_loss": trainer.final_test_loss, "final_test_acc": trainer.final_test_acc})
                writer.writerow(csv_dict)

        else:
            vis.show(lambda : trainer.trainModel(model, train_set, test_set, opts))
            csv_dict = opts.values()    
            csv_dict.update({"final_test_loss": trainer.final_test_loss, "final_test_acc": trainer.final_test_acc})
            writer.writerow(csv_dict)
    
    # Save weights (optional)
    if args.save_weights:
        torch.save(model.state_dict(), opts.output_path("weights.pt"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_weights", action="store_true")
    parser.add_argument("--grad_cam", action="store_true")
    main(parser.parse_args())