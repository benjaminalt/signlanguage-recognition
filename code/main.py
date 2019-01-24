import os
import torch
import hiddenlayer as hl
import csv
import argparse
import time

from train import ModelTrainer
import options
import logger

from torchvision import transforms
from datasets.SignMNISTDataset import SignMNISTDataset
from nets import *
from visualizer.Visualizer import Visualizer
from visualizer.GradCamVisualizer import GradCamVisualizer
from visualizer.ConvolutionVisualizer import ConvolutionVisualizer


def train(model, train_set, test_set, opts):
    """
    Train model, create visualization & document results in CSV file
    """

    print("Training model...")
    trainer = ModelTrainer()
    vis = Visualizer(opts)

    with open(opts.output_path("results.csv"), "w") as csv_file:
        fieldnames = opts.var_names()
        fieldnames.extend(["final_test_loss", "final_test_acc", "duration"])
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        if opts.gridSearch:
            for config in opts.iter():
                print("Testing {}".format(config))
                start_time = time.time()
                vis.show(lambda: trainer.trainModel(model, train_set, test_set, opts))
                csv_dict = opts.values()
                end_time = time.time()
                csv_dict.update({"final_test_loss": trainer.final_test_loss, "final_test_acc": trainer.final_test_acc, "duration": end_time - start_time})
                writer.writerow(csv_dict)
                csv_file.flush()

        else:
            start_time = time.time()
            vis.show(lambda: trainer.trainModel(model, train_set, test_set, opts))
            end_time = time.time()
            csv_dict = opts.values()
            csv_dict.update({"final_test_loss": trainer.final_test_loss, "final_test_acc": trainer.final_test_acc, "duration": end_time - start_time})
            writer.writerow(csv_dict)


def test(model, data_filepath, opts):
    """
    Test a given pytorch model by calculating the loss and accuracy
    """

    print("Evaluating model on {}...".format(data_filepath))
    test_set = SignMNISTDataset(opts, data_filepath)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=opts.batch_size, shuffle=opts.shuffleTestData, num_workers=1)
    _, loss, acc, _ = ModelTrainer.test(opts, test_loader, test_set, model, torch.nn.CrossEntropyLoss(), 0)
    print("Loss: {}".format(loss))
    print("Accuracy: {}".format(acc))


def visualize(model, data_filepath, opts):
    """
    Generate and store various visualizations for a given model
    """

    test_set = SignMNISTDataset(opts, data_filepath)
    # print("Generating GradCAM visualization...")
    # visualizer = GradCamVisualizer(opts)
    # for idx, sample in enumerate(test_set):
    #     output_dir = os.path.join(opts.output_path("gradcam"), str(idx))
    #     os.makedirs(output_dir)
    #     for n_layer in range(6):
    #         visualizer.visualize(model, n_layer, sample["image"], sample["label"], output_dir, "gradcam_{}".format(n_layer))
    # print("Done.")
    print("Visualizing filters...")
    visualizer = ConvolutionVisualizer(opts)
    output_dir = opts.output_path("filters")
    os.makedirs(output_dir)
    for n_layer in range(6):
        visualizer.visualize(model, n_layer, output_dir)
    

def main(args):
    """
    Main function of sign-language recognition
    :param args: Command line arguments
    :return: nothing
    """

    # Create default option parameters
    opts = options.Options()

    # Create output directory:
    os.makedirs(opts.output_dir(), exist_ok=True)

    # Log output and error messages
    logger.OutputLogger(opts)
    logger.ErrorLogger(opts)

    # Load a neural net model (different models are available in the sub-folder "nets")
    # Good choices:
    # CNN_3, dropout=0.90-0.95
    # CNN_5, dropout=0.95
    # CNN_12, dropout=0.95
    # CNN_13, dropout=0.95 <- Simple architecture: 6 CNN layers, fixed channel numbers, was even able to achieve 100% accuracy during one test
    # CNN_14, dropout=0.95
    model = CNNs.CNN_13(opts)

    # Optionally load previously calculated weights
    if args.weights is not None and os.path.isfile(args.weights):
        print("Loading model from {}".format(args.weights))
        model.load_state_dict(torch.load(args.weights, map_location="cpu"))

    model_graph_file = opts.output_path("model")

    # Generate model graph visualization
    print("Generating model graph visualization at " + opts.root_relpath(model_graph_file) + " ...")
    hl.build_graph(model, torch.zeros([1, 1, 28, 28])).save(model_graph_file)

    # Check if CUDA is used
    if opts.use_cuda:
        print("Using CUDA!")
        model = model.cuda()
    else:
        print("Not using CUDA!")

    if args.mode == "train":
        # Train model and save weights subsequently
        train_path = opts.data_path("sign_mnist_train.csv")
        test_path  = opts.data_path("sign_mnist_test.csv")
        #test_path  = opts.data_path("custom_dataset.csv")

        # Random additional data augmentation:
        # Note: Using more augmentations might require slightly more epochs for best results
        train_transform = transforms.Compose([
            #transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0),
            transforms.RandomResizedCrop(size=28, scale=(0.9, 1.0))#, # this works really well
            #transforms.RandomRotation(degrees=20)#,
            #transforms.RandomHorizontalFlip(p=0.2) # this might require an increased number of epochs
        ])

        train_set = SignMNISTDataset(opts, train_path, transform=train_transform)
        test_set  = SignMNISTDataset(opts, test_path)
        train(model, train_set, test_set, opts)
        weights_file = opts.output_path("weights.pt")
        print("Saving model weights at " + opts.root_relpath(weights_file) + " ...")
        torch.save(model.state_dict(), weights_file)

    elif args.mode == "test":
        # Test model
        if args.data is None or not os.path.isfile(args.data):
            raise ValueError("Invalid parameter 'data'")
        test(model, args.data, opts)

    elif args.mode == "visualize":
        # Visualize model
        if args.data is None or not os.path.isfile(args.data):
            raise ValueError("Invalid parameter 'data'")
        visualize(model, args.data, opts)

    else:
        raise ValueError("Invalid mode '{}' given by argument".format(args.mode))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", type=str, help="train|test|visualize")
    parser.add_argument("--weights", type=str, help="Path to pytorch (.pt) file containing the learned weights for initializing the model")
    parser.add_argument("--data", type=str, help="Path to CSV file containing the test data (in 'test' or 'visualize' mode only)")
    main(parser.parse_args())
