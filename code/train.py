import time

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam


def trainModel(model, train_set, test_set, batch_size, n_epochs, learning_rate=0.001, weight_decay=0.0001):
    print("===== HYPERPARAMETERS =====")
    print("batch_size=", batch_size)
    print("epochs=", n_epochs)
    print("learning_rate=", learning_rate)
    print("=" * 30)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader  = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)

    n_batches = len(train_loader)

    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_fn = torch.nn.CrossEntropyLoss()

    training_start_time = time.time()

    for epoch in range(n_epochs):

        # calculate test loss and accuracy
        print("[Test] Beginning with test with {} batches".format(len(test_loader)))
        test_starting_time = time.time()

        test_accuracy = 0
        test_loss = 0
        for i, data in enumerate(test_loader, 0):
            labels = data['label']
            inputs = data['image']
            inputs = inputs.type(torch.FloatTensor)

            outputs = model(inputs)
            loss = loss_fn(outputs, labels)

            values, pred_labels = outputs.max(dim=1)
            test_accuracy += labels.eq(pred_labels).sum().item()
            test_loss += loss.item()

        test_accuracy /= len(test_set.labels)
        test_loss /= len(test_loader)

        yield True, test_loss, test_accuracy, epoch

        print("[Test] Test finished, took {:.2f}s".format(time.time() - test_starting_time))

        running_loss = 0.0
        print_every = n_batches // 10
        start_time = time.time()
        total_train_loss = 0

        for i, data in enumerate(train_loader, 0):
            labels = data['label']
            inputs = data['image']
            inputs = inputs.type(torch.FloatTensor)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            total_train_loss += loss.item()

            if (i + 1) % (print_every + 1) == 0:
                print("Epoch {}, {:d}% \t running_train_loss: {:.2f} took: {:.2f}s".format(
                    epoch + 1, int(100 * (i + 1) / n_batches), running_loss / print_every, time.time() - start_time))

                values, pred_labels = outputs.max(dim=1)
                accuracy = labels.eq(pred_labels).float().mean()

                yield False, loss.item(), accuracy.item(), epoch + (i + 1) / n_batches
                running_loss = 0.0
                start_time = time.time()

        print("Training finished, took {:.2f}s".format(time.time() - training_start_time))
    return None
