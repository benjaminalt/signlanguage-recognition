import torch


class SimpleCNN(torch.nn.Module):

    def __init__(self):
        super(SimpleCNN, self).__init__()

        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=18, kernel_size=3, stride=1,
                                     padding=1)  # out: [18 x 28 x 28]
        self.relu1 = torch.nn.ReLU()
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # out: [18 x 14 x 14]
        self.net = torch.nn.Sequential(self.conv1, self.relu1, self.pool1)

        self.fc1 = torch.nn.Linear(in_features=18 * 14 * 14, out_features=64)  # out: [64 x 1]
        self.relu2 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(in_features=64, out_features=25)  # out: [25 x 1]

    def forward(self, input):
        output = self.net(input)
        output = output.view(-1, 18 * 14 * 14)
        output = self.fc1(output)
        output = self.relu2(output)
        output = self.fc2(output)
        return output
