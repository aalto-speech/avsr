# Author: David Harwath
import torch.nn as nn
import torch.nn.functional as F

class ConvX3AudioNet(nn.Module):
    def __init__(self, embedding_dim=1024, input_length=2048):
        super(ConvX3AudioNet, self).__init__()
        self.embedding_dim = embedding_dim
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(40, 5), stride=(1, 1), padding=(0, 2))
        self.conv2 = nn.Conv2d(64, 512, kernel_size=(1, 25), stride=(1, 1), padding=(0, 12))
        self.conv3 = nn.Conv2d(512, 1024, kernel_size=(1, 25), stride=(1, 1), padding=(0, 12))
        self.pool = nn.MaxPool2d(kernel_size=(1, 4), stride=(1, 2), padding=(0, 1))
        self.globalPool = nn.MaxPool2d(kernel_size=(1, input_length//4), stride=(1, 1), padding=(0, 0))

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = F.normalize(self.globalPool(x), dim=1)
        x = x.squeeze()
        return x


# Classifier version of the above net. This classifier is trained as a warmup
# for the above net.
class ConvX3AudioClassifierNet(nn.Module):
    def __init__(self, embedding_dim=1024, input_length=2048):
        super(ConvX3AudioClassifierNet, self).__init__()
        self.embedding_dim = embedding_dim
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(40, 5), stride=(1, 1), padding=(0, 2))
        self.conv2 = nn.Conv2d(64, 512, kernel_size=(1, 25), stride=(1, 1), padding=(0, 12))
        self.conv3 = nn.Conv2d(512, 1024, kernel_size=(1, 25), stride=(1, 1), padding=(0, 12))
        self.pool = nn.MaxPool2d(kernel_size=(1, 4), stride=(1, 2), padding=(0, 1))
        self.globalPool = nn.MaxPool2d(kernel_size=(1, input_length//4), stride=(1, 1), padding=(0, 0))
        self.fc = nn.Linear(1024, 205)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = F.normalize(self.globalPool(x), dim=1)
        x = x.squeeze()
        x = self.fc(x)
        return x
