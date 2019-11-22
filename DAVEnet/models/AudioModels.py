# Author: David Harwath
import torch.nn as nn
import torch.nn.functional as F


class DaveNet(nn.Module):
    def __init__(self, embedding_dim=1024):
        super(DaveNet, self).__init__()
        self.embedding_dim = embedding_dim
        self.batchnorm1 = nn.BatchNorm2d(1)
        self.conv1 = nn.Conv2d(1, 128, kernel_size=(40, 1), stride=(1, 1), padding=(0, 0))
        self.conv2 = nn.Conv2d(128, 256, kernel_size=(1, 11), stride=(1, 1), padding=(0, 5))
        self.conv3 = nn.Conv2d(256, 512, kernel_size=(1, 17), stride=(1, 1), padding=(0, 8))
        self.conv4 = nn.Conv2d(512, 512, kernel_size=(1, 17), stride=(1, 1), padding=(0, 8))
        self.conv5 = nn.Conv2d(512, embedding_dim, kernel_size=(1, 17), stride=(1, 1),
                               padding=(0, 8))
        self.pool = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.batchnorm1(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = F.relu(self.conv5(x))
        x = self.pool(x)
        x = x.squeeze(2)
        return x


class DaveNetClassifier(nn.Module):
    def __init__(self, embedding_dim=1024, input_length=2048):
        super(DaveNetClassifier, self).__init__()
        self.embedding_dim = embedding_dim
        self.input_length = input_length
        self.batchnorm1 = nn.BatchNorm2d(1)
        self.conv1 = nn.Conv2d(1, 128, kernel_size=(40, 1), stride=(1, 1), padding=(0, 0))
        self.conv2 = nn.Conv2d(128, 256, kernel_size=(1, 11), stride=(1, 1), padding=(0, 5))
        self.conv3 = nn.Conv2d(256, 512, kernel_size=(1, 17), stride=(1, 1), padding=(0, 8))
        self.conv4 = nn.Conv2d(512, 512, kernel_size=(1, 17), stride=(1, 1), padding=(0, 8))
        self.conv5 = nn.Conv2d(512, embedding_dim, kernel_size=(1, 17), stride=(1, 1),
                               padding=(0, 8))
        self.pool = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        self.fc = nn.Linear(self.embedding_dim * self.input_length // 2**4, 205)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.batchnorm1(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = F.relu(self.conv5(x))
        x = self.pool(x)
        x = x.squeeze()
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# ResDAVEnet has 1x9 convolutions instead of 3x3
def conv1x9(in_planes, out_planes, stride=1):
    """1x9 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=(1, 9), stride=stride,
                     padding=(0, 4), bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


# This BasicBlock is an adjusted version of the one defined in torch example implementation of ResNet:
# https://pytorch.org/docs/stable/_modules/torchvision/models/resnet.html
class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        norm_layer = nn.BatchNorm2d
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x9(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU()
        self.conv2 = conv1x9(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResDaveNet(nn.Module):

    def __init__(self, embedding_dim=1024):
        super(ResDaveNet, self).__init__()

        self._norm_layer = nn.BatchNorm2d
        self.inplanes = 128

        self.embedding_dim = embedding_dim
        self.conv1 = nn.Conv2d(1, 128, kernel_size=(40, 1), stride=(1, 1), padding=(0, 0))
        self.relu = nn.ReLU()
        self.batchnorm1 = self._norm_layer(self.inplanes)

        self.stack1 = self._make_residual_block(BasicBlock, 128, 2, stride=2)
        self.stack2 = self._make_residual_block(BasicBlock, 256, 2, stride=2)
        self.stack3 = self._make_residual_block(BasicBlock, 512, 2, stride=2)
        self.stack4 = self._make_residual_block(BasicBlock, 1024, 2, stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_residual_block(self, block, planes, blocks, stride=1):

        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def _forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.batchnorm1(x)

        x = self.stack1(x)
        x = self.stack2(x)
        x = self.stack3(x)
        x = self.stack4(x)

        x = x.squeeze(2)
        return x

    # Allow for accessing forward method in a inherited class
    forward = _forward
