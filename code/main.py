import os
import torch
import hiddenlayer as hl
import csv

from train import ModelTrainer
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
    hl.build_graph(model, torch.zeros([1,1,28,28])).save(os.path.join(opts.output_dir(), "model"))

    trainer = ModelTrainer()
    vis = Visualizer(opts)

    # Train, visualize & document results in CSV file
    with open(os.path.join(opts.output_dir(), "results.csv"), "w") as csv_file:
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

if __name__ == "__main__":
    main()
