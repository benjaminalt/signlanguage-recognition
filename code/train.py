import time

from torch.utils.data import DataLoader
from torch.optim import Adam
import torch


class ModelTrainer(object):
    def __init__(self):
        self.final_test_loss = 0
        self.final_test_acc = 0

    def trainModel(self, model, train_set, test_set, options):
        """
        Train a given model with passed data.
        This generator yields temporary training and testing progresses in order to plot them.
        """

        print("===== HYPERPARAMETERS =====")
        print("batch_size=", options.batch_size)
        print("epochs=", options.n_epochs)
        print("learning_rate=", options.learning_rate)
        print("=" * 30)

        # Load data from files
        train_loader = DataLoader(train_set, batch_size=options.batch_size, shuffle=options.shuffleTrainData, num_workers=4)
        test_loader = DataLoader(test_set, batch_size=options.batch_size, shuffle=options.shuffleTestData, num_workers=4)

        n_batches = len(train_loader)

        # Create optimizer and loss function
        optimizer = Adam(model.parameters(), lr=options.learning_rate, weight_decay=options.weight_decay)
        loss_fn = torch.nn.CrossEntropyLoss()

        # Start time measurement
        training_start_time = time.time()

        # Train for multiple epochs
        for epoch in range(options.n_epochs):

            # Calculate test loss and accuracy
            yield ModelTrainer.test(options, test_loader, test_set, model, loss_fn, epoch)

            print_every = n_batches // 10
            start_time = time.time()

            # Train net with all batches in the training data
            for i, data in enumerate(train_loader, 0):
                labels = data['label']
                inputs = data['image']
                if options.use_cuda:
                    inputs = inputs.type(torch.cuda.FloatTensor)
                    labels = labels.type(torch.cuda.LongTensor)
                else:
                    inputs = inputs.type(torch.FloatTensor)
                    labels = labels.type(torch.LongTensor)
                optimizer.zero_grad()

                model.train(True)
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()

                # Calculate and yield test results in regular intervals
                if (i + 1) % (print_every + 1) == 0:
                    values, pred_labels = outputs.max(dim=1)
                    accuracy = labels.eq(pred_labels).float().mean()

                    print("[Train] Epoch {}, {:d}% \t loss: {:.4f}, acc: {:.4f}, time: {:.2f}s".format(
                        epoch + 1, int(100 * (i + 1) / n_batches), loss.item(), accuracy, time.time() - start_time))

                    yield False, loss.item(), accuracy.item(), epoch + (i + 1) / n_batches
                    start_time = time.time()

        # Final test at the end
        _, self.final_test_loss, self.final_test_acc, _ = ModelTrainer.test(options, test_loader, test_set, model, loss_fn, epoch)
        print("Training finished, took {:.2f}s".format(time.time() - training_start_time))
        return None

    @staticmethod
    def test(options, loader, test_set, model, loss_fn, epoch):
        """
        Evaluate the nets loss and accuracy
        """

        print("[Test] Beginning with test with {} batches".format(len(loader)))
        test_starting_time = time.time()

        test_accuracy = 0
        test_loss = 0
        for i, data in enumerate(loader, 0):
            labels = data['label']
            inputs = data['image']
            if options.use_cuda:
                inputs = inputs.type(torch.cuda.FloatTensor)
                labels = labels.type(torch.cuda.LongTensor)
            else:
                inputs = inputs.type(torch.FloatTensor)
                labels = labels.type(torch.LongTensor)

            model.train(False)
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)

            values, pred_labels = outputs.max(dim=1)

            test_accuracy += labels.eq(pred_labels).sum().item()
            test_loss += loss.item()

        test_accuracy /= len(test_set.labels)
        test_loss /= len(loader)
        print("[Test] Test finished, loss: {:.4f}, acc: {:.4f}, time: {:.2f}s".format(test_loss, test_accuracy, time.time() - test_starting_time))
        return True, test_loss, test_accuracy, epoch
