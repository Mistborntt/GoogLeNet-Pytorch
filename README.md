# GoogLeNet-Pytorch
![GoogLeNet](https://badgen.net/badge/GoogLeNet/Pytorch/cyan?icon=github)

代码使用Pytorch构建GoogLeNet架构，Pytorch可从官网https://pytorch.org/get-started/locally/查看下载方式，
torchinfo可由pip install下载，主要用来显示网络具体架构。

'GoogLeNet.pdf'是原论文。

'GoogLeNet.png'展示了GoogLeNet的具体架构，主要亮点就是加入了Inception层。

'GoogLeNet参数.png'展示了每一层结构的具体参数，代码中会用到。

'GoogLeNet.py'中包含了四个类：'BasicConv2d'是基础的卷积层，'Inception'就是Inception层，'Auxclf'是辅助分类器，'GoogLeNet'就是GoogLeNet网络主体。
