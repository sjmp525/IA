'''VGG for CIFAR10. FC layers are removed.
(c) YANG, Wei
'''
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math
from utils.DF_ABNNlib import DA, RPReLU

__all__ = [
    'VGG', 'vgg_small_1w1a'
]


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
}


class VGG(nn.Module):

    def __init__(self, features, num_classes=1000):
        super(VGG, self).__init__()
        self.features = features
        self.bn2 = nn.BatchNorm1d(512 * block.expansion)
        self.nonlinear2 = nn.Hardtanh(inplace=True)
        self.classifier = nn.Linear(512, num_classes)
        self.bn3 = nn.BatchNorm1d(num_classes)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.bn2(x)
        x = self.nonlinear2(x)
        x = self.classifier(x)
        x = self.bn3(x)
        x = self.logsoftmax(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.Hardtanh(inplace=True)]
            else:
                layers += [conv2d, nn.Hardtanh(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'F': [128, 128, 'M', 512, 512, 'M'],
}



class VGG_SMALL_1W1A(nn.Module):

    def __init__(self, num_classes=10):
        super(VGG_SMALL_1W1A, self).__init__()
        self.conv0 = nn.Conv2d(3, 128, kernel_size=3, padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(128)
        self.conv1 = AdaBin_Conv2d(128, 128, kernel_size=3, padding=1, bias=False)
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm2d(128)
        # self.nonlinear = nn.ReLU(inplace=True)
        self.nonlinear = nn.Hardtanh(inplace=True)
        self.conv2 = AdaBin_Conv2d(128, 256, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = AdaBin_Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = AdaBin_Conv2d(256, 512, kernel_size=3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(512)
        self.conv5 = AdaBin_Conv2d(512, 512, kernel_size=3, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(512)
        self.fc = nn.Linear(512*4*4, num_classes)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, AdaBin_Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


    def forward(self, x):
        x = self.conv0(x)
        x = self.bn0(x)
        x = self.nonlinear(x)
        x = self.conv1(x)
        x = self.pooling(x)
        x = self.bn1(x)
        x = self.nonlinear(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.nonlinear(x)
        x = self.conv3(x)
        x = self.pooling(x)
        x = self.bn3(x)
        x = self.nonlinear(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.nonlinear(x)
        x = self.conv5(x)
        x = self.pooling(x)
        x = self.bn5(x)
        s0 = self.nonlinear(x)
        # s0 = self.pooling(s0)
        s1 = s0.view(s0.size(0), -1)
        x = self.fc(s1)
        return [s0, s1], x




def vgg_small_1w1a(**kwargs):
    model = VGG_SMALL_1W1A(**kwargs)
    return model
