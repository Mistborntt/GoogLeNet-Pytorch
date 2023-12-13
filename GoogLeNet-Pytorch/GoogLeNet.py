import torch
import torch.nn as nn
from torchinfo import summary

class BasicConv2d(nn.Module):
    def __init__(self, inplanes, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(inplanes, out_channels, bias=False, **kwargs),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Inception(nn.Module):
    def __init__(self, inplanes, ch1x1, ch3x3reduce, ch3x3, ch5x5reduce, ch5x5, pool_proj):
        super(Inception, self).__init__()

        self.branch1 = BasicConv2d(inplanes, ch1x1, kernel_size=1)
        self.branch2 = nn.Sequential(
            BasicConv2d(inplanes, ch3x3reduce, kernel_size=1),
            BasicConv2d(ch3x3reduce, ch3x3, kernel_size=3, padding=1)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(inplanes, ch5x5reduce, kernel_size=1),
            BasicConv2d(ch5x5reduce, ch5x5, kernel_size=5, padding=2)
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(inplanes, pool_proj, kernel_size=1)
        )

    def forward(self, x):
        branchh1 = self.branch1(x)
        branchh2 = self.branch2(x)
        branchh3 = self.branch3(x)
        branchh4 = self.branch4(x)

        output = [branchh1, branchh2, branchh3, branchh4]

        return torch.cat(output, 1)  # 按照通道数拼接


# 辅助分类器
class AuxClf(nn.Module):
    def __init__(self, inplanes, num_classes, **kwargs):
        super(AuxClf, self).__init__()

        self.feature_ = nn.Sequential(
            nn.AvgPool2d(kernel_size=5, stride=3),
            BasicConv2d(inplanes, 128, kernel_size=1)
        )
        self.clf_ = nn.Sequential(
            nn.Linear(4*4*128, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.7),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.feature_(x)
        x = x.view(-1, 4*4*128)  # 将x重新排列为一个二维张量，自动计算第一个维度的大小，以确保总元素数保持不变
        x = self.clf_(x)

        return x


class GoogLeNet(nn.Module):
    def __init__(self, num_classes: int = 1000):
        super(GoogLeNet, self).__init__()

        self.conv1 = BasicConv2d(3, 64, kernel_size=7, stride=2, padding=3)  # 输出112x112x64
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)  # (112 - 3) / 2 + 1 = 55.5，向上取整，输出56x56x64

        self.conv2 = BasicConv2d(64, 64, kernel_size=1, stride=1)
        self.conv3 = BasicConv2d(64, 192, kernel_size=3, stride=1, padding=1)  # 输出56x56x192
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)  # 输出28x28x192

        self.inception3a = Inception(192,64,96, 128, 16, 32, 32)  # 输出28x28x256
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)  # 输出28x28x480
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)  # 输出14x14x480

        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)  # 输出14x14x512
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)  # 输出14x14x512
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)  # 输出14x14x512
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)  # 输出14x14x528
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)  # 输出14x14x832
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)  # 输出7x7x832

        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)  # 输出7x7x832
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)  # 输出7x7x1024

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # 输出1x1x1024
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, num_classes)

        self.aux1 = AuxClf(512, num_classes)
        self.aux2 = AuxClf(528, num_classes)

    def forward(self, x):
        x = self.maxpool1(self.conv1(x))

        x = self.maxpool2(self.conv3(self.conv2(x)))

        x = self.maxpool3(self.inception3b(self.inception3a(x)))

        x = self.inception4a(x)
        aux1 = self.aux1(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        aux2 = self.aux2(x)
        x = self.maxpool4(self.inception4e(x))

        x = self.inception5b(self.inception5a(x))

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        return x, aux1, aux2

if __name__ == '__main__':
    data = torch.ones(20, 3, 224, 224)
    net = GoogLeNet()
    x, aux1, aux2 = net(data)
    for i in [x, aux1, aux2]:
        print(i.shape)
    summary(net, (20, 3, 224, 224), device='cpu')