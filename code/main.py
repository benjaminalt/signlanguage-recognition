import os
import torch
import hiddenlayer as hl
import csv
import argparse

from train import ModelTrainer
import options
import logger

from datasets.SignMNISTDataset import SignMNISTDataset
from nets import *
from visualizer.Visualizer import Visualizer
from visualizer.GradCamVisualizer import GradCamVisualizer

def grad_cam(model, img, label, opts):
    print("Generating GradCAM visualization...")
    visualizer = GradCamVisualizer(opts)
    for n_layer in range(6):
        visualizer.visualize(model, n_layer, img, label)

def train(model, train_set, test_set, opts):
    """
    Train model, create visualization & document results in CSV file
    """
    print("Training model...")
    trainer = ModelTrainer()
    vis = Visualizer(opts)
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
                csv_file.flush()

        else:
            vis.show(lambda : trainer.trainModel(model, train_set, test_set, opts))
            csv_dict = opts.values()    
            csv_dict.update({"final_test_loss": trainer.final_test_loss, "final_test_acc": trainer.final_test_acc})
            writer.writerow(csv_dict)

def test(model, data_filepath, opts):
    print("Evaluating model on {}...".format(data_filepath))
    test_set  = SignMNISTDataset(opts, data_filepath)
    trainer = ModelTrainer()
    test_loader  = torch.utils.data.DataLoader(test_set, batch_size=opts.batch_size, shuffle=opts.shuffleTestData, num_workers=1)
    _, loss, acc, _ = trainer.test(opts, test_loader, test_set, model, torch.nn.CrossEntropyLoss(), 0)
    print("Loss: {}".format(loss))
    print("Accuracy: {}".format(acc))


def main(args):
    opts = options.Options()

    # Create output directory:
    os.makedirs(opts.output_dir(), exist_ok=True)

    # Log output and error messages
    logger.OutputLogger(opts)
    logger.ErrorLogger(opts)

    train_path = opts.data_path("sign_mnist_train.csv")
    test_path = opts.data_path("sign_mnist_test.csv")

    train_set = SignMNISTDataset(opts, train_path)
    test_set  = SignMNISTDataset(opts, test_path)

    model = CNNs.CNN_5(opts)

    if args.weights is not None and os.path.isfile(args.weights):
        print("Loading model from {}".format(args.weights))
        model.load_state_dict(torch.load(args.weights))
    
    model_graph_file = opts.output_path("model")
    print("Generating model graph visualization at " + opts.root_relpath(model_graph_file) + " ...")
    hl.build_graph(model, torch.zeros([1,1,28,28])).save(model_graph_file)

    # Check if using cuda:
    if opts.use_cuda:
        print("Using CUDA!")
        model = model.cuda()
    else:
        print("Not using CUDA!")

    if args.mode == "train":
        train(model, train_set, test_set, opts)
        # Save weights
        weights_file = opts.output_path("weights.pt")
        print("Saving model weights at " + opts.root_relpath(weights_file) + " ...")
        torch.save(model.state_dict(), weights_file)
    elif args.mode == "test":
        if args.data is None or not os.path.isfile(args.data):
            raise ValueError("Invalid parameter 'data'")
        test(model, args.data, opts)

    if args.grad_cam:
        grad_cam(model, train_set[0]["image"], train_set[0]["label"], opts)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", type=str, help="train|test")
    parser.add_argument("--weights", type=str, help="Path to pytorch (.pt) file containing the learned weights for initializing the model")
    parser.add_argument("--data", type=str, help="Path to CSV file containing the test data")
    parser.add_argument("--grad_cam", action='store_true')
    main(parser.parse_args())