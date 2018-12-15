import torch

class DeepCNN(torch.nn.Module):

    def __init__(self, dropout_probability=0.3):
        super(DeepCNN, self).__init__()

        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=18, kernel_size=3, stride=1,
                                     padding=1)  # out: [18 x 28 x 28]
        self.relu1 = torch.nn.ReLU()
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # out: [18 x 14 x 14]
        
        self.conv2 = torch.nn.Conv2d(in_channels=18, out_channels=18, kernel_size=3, stride=1,
                                     padding=1)  # out: [18 x 14 x 14]
        self.relu2 = torch.nn.ReLU()
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # out: [18 x 7 x 7]
        
        self.net = torch.nn.Sequential(self.conv1, self.relu1, self.pool1, self.conv2, self.relu2, self.pool2)

        self.dropout1 = torch.nn.Dropout(dropout_probability)
        self.fc1 = torch.nn.Linear(in_features=18 * 7 * 7, out_features=64)  # out: [64 x 1]
        self.relu3 = torch.nn.ReLU()
        self.dropout2 = torch.nn.Dropout(dropout_probability)
        self.fc2 = torch.nn.Linear(in_features=64, out_features=25)  # out: [25 x 1]

    def forward(self, input):
        output = self.net(input)
        output = output.view(-1, 18 * 7 * 7)
        output = self.dropout1(output)
        output = self.fc1(output)
        output = self.relu3(output)
        output = self.dropout2(output)
        output = self.fc2(output)
        return output