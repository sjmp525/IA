import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.DF_ABNNlib import DA, RPReLU, IRConv2d, BinarizeConv2d_RBNN, BinarizeConv2d_ReCU, BinarizeConv2d_BiPer, Hist_Show
import torch.nn.init as init
import math
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
import numpy as np
import torchvision
import torch.utils.model_zoo as model_zoo

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

class BasicBlock_1w1a_18(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, a_bit=32, w_bit=32, option='A'):
        super(BasicBlock_1w1a_18, self).__init__()

        self.conv1 = DA(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, a_bit=a_bit, w_bit = w_bit)
        self.bn1 = nn.BatchNorm2d(planes)
        self.nonlinear1 = RPReLU(planes)

        self.conv2 = DA(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.nonlinear2 = RPReLU(planes)

        self.shortcut = nn.Sequential()

        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     BinarizeConv2d_BiPer(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.nonlinear1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(x)
        out = self.nonlinear2(out)
        return out

class BasicBlock_1w1a_18_ti(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock_1w1a_18_ti, self).__init__()

        self.conv1 = DA(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.nonlinear1 = RPReLU(planes)

        self.conv2 = DA(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.nonlinear2 = RPReLU(planes)


    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.nonlinear1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.nonlinear2(out)
        return out

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = DA(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.nonlinear1 = RPReLU(planes)
        self.conv2 = DA(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.nonlinear2 = RPReLU(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     DA(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = self.nonlinear1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.nonlinear2(out)
        return out

class BasicBlock_t(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock_t, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet_t(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet_t, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)
        # self.linear = nn.Linear(512 * 8*8, num_classes)
        # self.bn2 = nn.BatchNorm1d(512 * block.expansion)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def get_feat_modules(self):
        feat_m = nn.ModuleList([])
        feat_m.append(self.conv1)
        feat_m.append(self.bn1)
        feat_m.append(self.relu)
        feat_m.append(self.layer1)
        feat_m.append(self.layer2)
        feat_m.append(self.layer3)
        feat_m.append(self.layer4)
        feat_m.append(self.linear)
        return feat_m

    def forward(self, x, out_feature=False):
        t0 = self.relu(self.bn1(self.conv1(x)))
        t1 = self.layer1(t0)
        t2 = self.layer2(t1)
        t3 = self.layer3(t2)
        t4 = self.layer4(t3)

        out1 = F.avg_pool2d(t4, 4)
        t5 = out1.view(out1.size(0), -1)
        # out = self.bn2(t5)
        out = self.linear(t5)
        if out_feature == False:
            return t4, out
        else:
            return out

class ResNet(nn.Module):

    def __init__(self, depth, num_filters, block, num_class=100):
        super(ResNet, self).__init__()
        # Model type specifies number of layers for CIFAR-100 model
        assert (depth - 2) % 6 == 0, 'When use basicblock, depth should be 6n+2, e.g. 20, 32, 44, 56, 110, 1202'
        n = (depth - 2) // 6

        self.inplanes = num_filters[0]
        self.conv1 = nn.Conv2d(3, num_filters[0], kernel_size=3, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters[0])
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, num_filters[1], n)
        self.layer2 = self._make_layer(block, num_filters[2], n, stride=2)
        self.layer3 = self._make_layer(block, num_filters[3], n, stride=2)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(num_filters[3] * block.expansion, num_class)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.normal_(m.bias, mean=0.0, std=0.01)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = list([])
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, embed=True):
        t0 =  self.relu(self.bn1(self.conv1(x)))

        t1 = self.layer1(t0)  # 32x32
        t2 = self.layer2(t1)  # 16x16
        t3 = self.layer3(t2)  # 8x8

        x = self.avgpool(t3)
        t4 = torch.flatten(x, 1) # 将第一个以后的维度展平，即将（batch，channel，height，width展开为batch，channel*height*width）
        logits = self.fc(t4)

        if embed:
            return [t0, t1, t2, t3, t4], logits
        else:
            return logits

class ResNet_s_18(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, a_bit=32, w_bit=32):
        super(ResNet_s_18, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.nonlinear1 = RPReLU(64)

        self.layer1 = self._make_layer(block, a_bit, w_bit, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, a_bit, w_bit, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, a_bit, w_bit, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, a_bit, w_bit, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)
        self.bn2 = nn.BatchNorm1d(512 * block.expansion)

        self.apply(_weights_init)
    def _make_layer(self, block, a_bit, w_bit, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, a_bit, w_bit))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, out_feature=False):
        s0 =(self.bn1(self.conv1(x)))
        s1 = self.layer1(s0)
        s2 = self.layer2(s1)
        s3 = self.layer3(s2)
        s4 = self.layer4(s3)

        out = F.avg_pool2d(s4, 4)
        s5 = out.view(out.size(0), -1)
        out = self.bn2(s5)
        out = self.linear(out)
        look = F.softmax(out, dim=1)
        if out_feature == False:
            return s4, out
        else:
            return out, s5
# Caltech101所用
class ResNet_s_18_cal(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet_s_18_cal, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.nonlinear1 = RPReLU(64)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)
        # self.bn2 = nn.BatchNorm1d(512 * block.expansion)

        self.apply(_weights_init)
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, out_feature=False):
        s0 =self.nonlinear1(self.bn1(self.conv1(x)))
        s0 = self.maxpool(s0)

        s1 = self.layer1(s0)
        s2 = self.layer2(s1)
        s3 = self.layer3(s2)
        s4 = self.layer4(s3)

        out = F.avg_pool2d(s4, 4)
        s5 = out.view(out.size(0), -1)
        out = self.linear(s5)
        look = F.softmax(out, dim=1)
        if out_feature == False:
            return s5, out
        else:
            return out, s5

class ResNet_s_18_ti(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet_s_18_ti, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.nonlinear1 = RPReLU(64)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512 * block.expansion, num_classes) # CIFAR10
        # self.linear = nn.Linear(512 * 16, num_classes) # Caltech101
        self.bn2 = nn.BatchNorm1d(512 * block.expansion) # CIFAR10
        # self.bn2 = nn.BatchNorm1d(512 * 16) # Caltech101

        self.apply(_weights_init)
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, out_feature=False):
        s0 =self.nonlinear1((self.bn1(self.conv1(x))))
        s1 = self.layer1(s0)
        s2 = self.layer2(s1)
        s3 = self.layer3(s2)
        s4 = self.layer4(s3)

        # out = F.avg_pool2d(s4, 4)
        out = self.avgpool(s4)
        s5 = out.view(out.size(0), -1)
        out = self.bn2(s5)
        out = self.linear(out)
        look = F.softmax(out, dim=1)
        # out = F.softmax(out, dim=1)
        if out_feature == False:
            return s4, out
        else:
            return out, s5

def _weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class ResNet_20(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet_20, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False) # CIFAR-10和CIFAR-100的卷积核尺寸大小设置为3；caltech101的卷积核大小设置为5
        self.bn1 = nn.BatchNorm2d(16)
        # self.nonlinear1 = RPReLU(16)

        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1e-8)
                m.bias.data.zero_()
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight)
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        # s0 = F.hardtanh(self.bn1(self.conv1(x)))
        s0 = (self.bn1(self.conv1(x)))
        s1 = self.layer1(s0)
        s2 = self.layer2(s1)
        s3 = self.layer3(s2)
        s4 = F.avg_pool2d(s3, s3.size()[3])
        s5 = s4.view(s4.size(0), -1)
        # s6 = self.bn2(s5)
        out = self.linear(s5)
        look = F.softmax(out , dim=1)
        return s5, out

    # def forward(self, x):
    #     s0 = F.relu(self.bn1(self.conv1(x)))
    #     s1 = self.layer1(s0)
    #     s2 = self.layer2(s1)
    #     s3 = self.layer3(s2)
    #     out = F.avg_pool2d(s3, s3.size()[3])
    #     s4 = out.view(out.size(0), -1)
    #     # out = self.bn2(s4)
    #     out = self.linear(s4)
    #     return [s0, s1, s2, s3, s4], out

class VGG_SMALL_1W1A(nn.Module):

    def __init__(self, num_classes=10):
        super(VGG_SMALL_1W1A, self).__init__()
        self.conv0 = nn.Conv2d(3, 128, kernel_size=3, padding=2, bias=False)
        self.bn0 = nn.BatchNorm2d(128)
        self.conv1 = DA(128, 128, kernel_size=3, padding=1, bias=False)
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm2d(128)
        self.nonlinear = nn.Hardtanh(inplace=True)
        self.conv2 = DA(128, 256, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = DA(256, 256, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = DA(256, 512, kernel_size=3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(512)
        self.conv5 = DA(512, 512, kernel_size=3, padding=1, bias=False)
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
            elif isinstance(m, DA):
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
        s1 = self.nonlinear(x)
        x = self.conv1(s1)
        x = self.pooling(x)
        x = self.bn1(x)
        s2 = self.nonlinear(x)
        x = self.conv2(s2)
        x = self.bn2(x)
        s3 = self.nonlinear(x)
        x = self.conv3(s3)
        x = self.pooling(x)
        x = self.bn3(x)
        s4 = self.nonlinear(x)
        x = self.conv4(s4)
        x = self.bn4(x)
        s5 = self.nonlinear(x)
        x = self.conv5(s5)
        x = self.pooling(x)
        x = self.bn5(x)
        s6 = self.nonlinear(x)

        s9 = s6.view(s6.size(0), -1)
        x = self.fc(s9)
        return s9, x


def conv3x3Binary(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return DA(in_planes, out_planes, kernel_size=3, stride=stride,
                    padding=1, bias=False)

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def resnet18_1w1a(num_classes=10, a_bit=32, w_bit=32):
    return ResNet_s_18(BasicBlock_1w1a_18, [2, 2, 2, 2], num_classes, a_bit, w_bit)

def resnet20_1w1a(num_classes=10):
    return ResNet_20(BasicBlock, [3, 3, 3], num_classes)

def resnet34_1w1a(num_classes=10):
    return ResNet_s_18(BasicBlock_1w1a_18, [3, 4, 6, 3], num_classes)

def ResNet34_8x(num_classes=10):
    return ResNet_t(BasicBlock_t, [3, 4, 6, 3], num_classes)

def vgg_small_1w1a(**kwargs):
    model = VGG_SMALL_1W1A(**kwargs)
    return model

def resnet18_1w1a_ti(num_classes=10):
    return ResNet_s_18_ti(BasicBlock_1w1a_18_ti, [2, 2, 2, 2], num_classes)

def resnet18_1w1a_cal(num_classes=10):
    return ResNet_s_18_cal(BasicBlock_1w1a_18, [2, 2, 2, 2], num_classes)

def resnet34_1w1a_cal(num_classes=10):
    return ResNet_s_18_cal(BasicBlock_1w1a_18, [3, 4, 6, 3], num_classes)