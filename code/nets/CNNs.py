import torch

class Conv(torch.nn.Module):

    def __init__(self, in_ch, out_ch, kernel=3, stride_=1, pad=1, pool=False, batchnorm=False):
        super(Conv, self).__init__()
        
        self.conv = torch.nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel, stride=stride_, padding=pad)
        self.bn = torch.nn.BatchNorm2d(out_ch)
        self.relu = torch.nn.ELU()
        self.basic_net = torch.nn.Sequential(self.conv, self.bn, self.relu) if batchnorm else torch.nn.Sequential(self.conv, self.relu)
        if pool:
            self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
            self.net = torch.nn.Sequential(self.basic_net, self.pool)
        else:
            self.net = self.basic_net
 
    def forward(self, x):
        return self.net(x)

class Flatten(torch.nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size()[0], -1)

class FC(torch.nn.Module):

    def __init__(self, in_size, out_size, dropout=0.0):
        super(FC, self).__init__()
        
        self.fc = torch.nn.Linear(in_features=in_size, out_features=out_size)
        self.relu = torch.nn.ELU()
        if (dropout > 0):
            self.dropout = torch.nn.Dropout(dropout)
            self.net = torch.nn.Sequential(self.fc, self.relu, self.dropout)
        else:
            self.net = torch.nn.Sequential(self.fc, self.relu)
    
    def forward(self, x):
        return self.net(x)


class CNN_1(torch.nn.Module):

    def __init__(self, options):
        super(CNN_1, self).__init__()
        
        # input: [1 x 28 x 28]
        self.conv1 = Conv(in_ch=1, out_ch=8, pool=False) # out: [8 x 28 x 28]
        self.conv2 = Conv(in_ch=8, out_ch=16, pool=True) # out: [16 x 14 x 14]
        self.conv3 = Conv(in_ch=16, out_ch=32, pool=True) # out: [32 x 7 x 7]
        self.conv4 = Conv(in_ch=32, out_ch=64, pool=False) # out: [64 x 7 x 7]
        self.conv5 = Conv(in_ch=64, out_ch=128, pool=True) # out: [128 x 3 x 3]
        self.conv_net = torch.nn.Sequential(self.conv1, self.conv2, self.conv3, self.conv4, self.conv5)

        self.flatten = Flatten()
        self.fc = FC(in_size=128 * 3 * 3, out_size=128, dropout=0.5) # out: [128 x 1]
        self.out = torch.nn.Linear(in_features=128, out_features=25) # out: [25 x 1]
        self.net = torch.nn.Sequential(self.conv_net, self.flatten, self.fc, self.out)

    def forward(self, x):
        return self.net(x)

class CNN_2(torch.nn.Module):

    def __init__(self, options):
        super(CNN_2, self).__init__()
        
        # input: [1 x 28 x 28]
        self.conv1 = Conv(in_ch=1, out_ch=64, pool=True) # out: [64 x 14 x 14]
        self.conv2 = Conv(in_ch=64, out_ch=64, pool=True) # out: [64 x 7 x 7]
        self.conv3 = Conv(in_ch=64, out_ch=64, pool=True) # out: [64 x 3 x 3]
        self.conv_net = torch.nn.Sequential(self.conv1, self.conv2, self.conv3)

        self.flatten = Flatten()
        self.fc = FC(in_size=64 * 3 * 3, out_size=128, dropout=0.5) # out: [128 x 1]
        self.out = torch.nn.Linear(in_features=128, out_features=25) # out: [25 x 1]
        self.net = torch.nn.Sequential(self.conv_net, self.flatten, self.fc, self.out)

    def forward(self, x):
        return self.net(x)


class CNN_3(torch.nn.Module):

    def __init__(self, options):
        super(CNN_3, self).__init__()
        
        # input: [1 x 28 x 28]
        self.conv1 = Conv(in_ch=1, out_ch=128, pool=False) # out: [128 x 28 x 28]
        self.conv2 = Conv(in_ch=128, out_ch=64, pool=True) # out: [64 x 14 x 14]
        
        self.conv3 = Conv(in_ch=64, out_ch=128, pool=False) # out: [128 x 14 x 14]
        self.conv4 = Conv(in_ch=128, out_ch=64, pool=True) # out: [64 x 7 x 7]
        
        self.conv5 = Conv(in_ch=64, out_ch=128, pool=False) # out: [128 x 7 x 7]
        self.conv6 = Conv(in_ch=128, out_ch=64, pool=True) # out: [64 x 3 x 3]
        
        self.conv_net = torch.nn.Sequential(self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.conv6)

        self.flatten = Flatten()
        self.fc = FC(in_size=64 * 3 * 3, out_size=128, dropout=0.2) # out: [128 x 1]
        self.out = torch.nn.Linear(in_features=128, out_features=25) # out: [25 x 1]
        self.net = torch.nn.Sequential(self.conv_net, self.flatten, self.fc, self.out)

    def forward(self, x):
        return self.net(x)

class CNN_4(torch.nn.Module):

    def __init__(self, options):
        super(CNN_4, self).__init__()
        
        # input: [1 x 28 x 28]
        self.conv1 = Conv(in_ch=1, out_ch=128, pool=False) # out: [128 x 28 x 28]
        self.conv2 = Conv(in_ch=128, out_ch=64, pool=True) # out: [64 x 14 x 14]
        
        self.conv3 = Conv(in_ch=64, out_ch=128, pool=False) # out: [128 x 14 x 14]
        self.conv4 = Conv(in_ch=128, out_ch=64, pool=True) # out: [64 x 7 x 7]
        
        self.conv5 = Conv(in_ch=64, out_ch=128, pool=False) # out: [128 x 7 x 7]
        self.conv6 = Conv(in_ch=128, out_ch=64, pool=True) # out: [64 x 3 x 3]
        
        self.conv_net = torch.nn.Sequential(self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.conv6)

        self.flatten = Flatten()
        self.fc = FC(in_size=64 * 3 * 3, out_size=256, dropout=0.5) # out: [256 x 1]
        self.out = torch.nn.Linear(in_features=256, out_features=25) # out: [25 x 1]
        self.net = torch.nn.Sequential(self.conv_net, self.flatten, self.fc, self.out)

    def forward(self, x):
        return self.net(x)

class CNN_5(torch.nn.Module):

    def __init__(self, options):
        super(CNN_5, self).__init__()
        
        bn = options.use_batchnorm_cnn

        # input: [1 x 28 x 28]
        self.conv1 = Conv(in_ch=1, out_ch=128, pool=False, batchnorm=bn) # out: [128 x 28 x 28]
        self.conv2 = Conv(in_ch=128, out_ch=64, pool=True, batchnorm=bn) # out: [64 x 14 x 14]
        
        self.conv3 = Conv(in_ch=64, out_ch=128, pool=False, batchnorm=bn) # out: [128 x 14 x 14]
        self.conv4 = Conv(in_ch=128, out_ch=64, pool=True, batchnorm=bn) # out: [64 x 7 x 7]
        
        self.conv5 = Conv(in_ch=64, out_ch=128, pool=False, batchnorm=bn) # out: [128 x 7 x 7]
        self.conv6 = Conv(in_ch=128, out_ch=64, pool=True, batchnorm=bn) # out: [64 x 3 x 3]
        
        self.conv_net = torch.nn.Sequential(self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.conv6)
        self.features = self.conv_net

        self.flatten = Flatten()
        self.fc = FC(in_size=64 * 3 * 3, out_size=256, dropout=0.95) # out: [256 x 1]
        self.out = torch.nn.Linear(in_features=256, out_features=25) # out: [25 x 1]
        self.classifier = torch.nn.Sequential(self.flatten, self.fc, self.out)
        self.net = torch.nn.Sequential(self.conv_net, self.flatten, self.fc, self.out)

    def forward(self, x):
        feature_activation = self.features(x)
        classification = self.classifier(feature_activation)
        return classification
        #return self.net(x)
