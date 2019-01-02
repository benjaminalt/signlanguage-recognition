import torch

class DeepCNN(torch.nn.Module):

    def __init__(self, options):
        super(DeepCNN, self).__init__()
        self.opts = options

        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1,
                                     padding=1)  # out: [8 x 28 x 28]
        self.conv1_bn = torch.nn.BatchNorm2d(8)
        self.relu1 = torch.nn.ELU()
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # out: [8 x 14 x 14]
        
        self.conv2 = torch.nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1,
                                     padding=1)  # out: [16 x 14 x 14]
        self.conv2_bn = torch.nn.BatchNorm2d(16)
        self.relu2 = torch.nn.ELU()
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # out: [16 x 7 x 7]
        
        self.conv3 = torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1,
                                     padding=1)  # out: [32 x 7 x 7]
        self.conv3_bn = torch.nn.BatchNorm2d(32)
        self.relu3 = torch.nn.ELU()
        #self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # out: [16 x 7 x 7]

        if options.use_batchnorm_cnn:
            self.net = torch.nn.Sequential(self.conv1, self.conv1_bn, self.relu1, self.pool1,
                                           self.conv2, self.conv2_bn, self.relu2, self.pool2,
                                           self.conv3, self.conv3_bn, self.relu3)
        else:
            self.net = torch.nn.Sequential(self.conv1, self.relu1, self.pool1, self.conv2, self.relu2, self.pool2, self.conv3, self.relu3)

        self.dropout1 = torch.nn.Dropout(options.dropout_probability_1)
        self.fc1 = torch.nn.Linear(in_features=32 * 7 * 7, out_features=64)  # out: [64 x 1]
        self.fc1_bn = torch.nn.BatchNorm1d(64)
        self.relu3 = torch.nn.ELU() # TODO: Check variable name. This line redefines self.relu3 (should not change anything but may be misleading)

        self.dropout2 = torch.nn.Dropout(options.dropout_probability_2)
        self.fc2 = torch.nn.Linear(in_features=64, out_features=32)  # out: [32 x 1]
        self.fc2_bn = torch.nn.BatchNorm1d(32)
        self.relu4 = torch.nn.ELU()

        self.dropout3 = torch.nn.Dropout(options.dropout_probability_3)
        self.fc3 = torch.nn.Linear(in_features=32, out_features=25)  # out: [25 x 1]

    def forward(self, input):
        output = self.net(input)
        output = output.view(-1, 32 * 7 * 7)
        output = self.dropout1(output)
        output = self.fc1(output)

        if self.opts.use_batchnorm_linear:
            output = self.fc1_bn(output)

        output = self.relu3(output)
        output = self.dropout2(output)
        output = self.fc2(output)

        if self.opts.use_batchnorm_linear:
            output = self.fc2_bn(output)

        output = self.relu4(output)
        output = self.dropout3(output)
        output = self.fc3(output)
        return output