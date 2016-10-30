---
layout:     post
title:      "Classification in Remote Sensing Optical Images by CNNs"
subtitle:   "CNN在光学遥感图像上的应用"
date:       2016-06-12 10:00:00
author:     "Ldy"
header-img: "img/tag-bg1.jpg"
tags:
    - CNN
    - Deep Learning
---

<center>
![](http://7xritj.com1.z0.glb.clouddn.com/16-6-11/17213122.jpg)
</center>
<!--more-->

## CNN简介

从06年开始，深度结构学习方法（深度学习或者分层学习方法）作为机器学习领域的新的研究方向出现。由于芯片处理性能的巨大提升，数据爆炸性增长，在过去的短短几年时间，深度学习技术得到快速发展，已经深深的影响了学术领域，其研究涉及的应用领域包括计算机视觉、语音识别、对话语音识别、图像特征编码、语意表达分类、自然语言理解、手写识别、音频处理、信息检索、机器人学。
![](http://7xritj.com1.z0.glb.clouddn.com/16-5-22/96930177.jpg)
![](http://7xritj.com1.z0.glb.clouddn.com/16-5-22/75458021.jpg)

由于深度学习在众多领域表现比较好的性能，越来越多的学术机构把目光投入深度学习领域。今年来活跃在机器学习领域的研究机构包括众多高校比如斯坦福，伯克利，还有一些企业例如Google，IBM 研究院，微软研究院，FaceBook，百度等等。

### 神经网络

人工神经网络是一种模仿生物神经网络(动物的中枢神经系统，特别是大脑)的结构和功能的数学模型或计算模型，简单结构如下图所示，包含了输入层，隐含层，和输出层，其中隐含层可能有多层。在神经网络中每个神经元都和它前一层的所有节点相连接，称之为全连接，其中每个连接以一定的权值相连接，**网络训练的过程就是得到权值的过程**。
![](http://7xritj.com1.z0.glb.clouddn.com/16-5-22/86340649.jpg)

不管是机器学习还是深度学习实际上都是在解决分类问题，当数据线性可分时，一个sigmoid函数就可以把数据分开，如下图所示，其中两类数据是线性可分的，我们只需要神经网络的输入和输出层就可以把两类数据分开，其中黄色的连线表示权值为负，蓝色的连线表示权值为正，连线的粗细表示权值的绝对值大小。
![](http://7xritj.com1.z0.glb.clouddn.com/16-5-22/47661239.jpg)

如下图，当原本的数据不可分时，我们就需要对数据进行一些非线性的变化，使得数据可分，而神经网络中的隐含层的作用就是对线性不可分的数据进行非线性变化，下图中包含了4个隐含层节点，数据被正确的分类。
![](http://7xritj.com1.z0.glb.clouddn.com/16-5-22/28174683.jpg)

### 卷积神经网络(CNN)

一维的CNN如下所示，和人工神经网络相比，CNN中的卷积层只与前一层节点的部分节点相连，称为局部连接，且卷积层中的每个神经元的权值相等，这一属性称为权值共享。卷积神经网络为什么有卷积两个字，就是因为这两个属性：局部连接，权值相等，具体原因可参考[http://colah.github.io/posts/2014-07-Understanding-Convolutions/](http://colah.github.io/posts/2014-07-Understanding-Convolutions/)。下图中的max层成为池化层(pooling),下图为max pooling ，就是对两个神经元的输出取其中的较大值。池化操作能够降低特征的维度(相比使用所有提取得到的特征)，同时还会改善结果(不容易过拟合)，池化单元也具有一定的平移不变性。下图中的B层为第二层卷积层卷积层，F层为全连接层，也就是上面所说的人工神经网络。

![](http://7xritj.com1.z0.glb.clouddn.com/16-5-22/35987179.jpg)

二维卷积神经网络如下所示，二维数据的输入可以看成是一张图像的每个像素值，**卷积层看做是一个滤波器对图像提取特征**，max pooling层相当于对图像进行更高维的抽象，然后后面连接全连接层(也就是传统的人工神经网络)进行分类。所以总的说来，利用CNN进行图像处理就是前面的卷积层对图像进行特征提取，经过学习提取出利于图像分类的特征，然后对提取出的特征利用人工神经网络进行分类。
![](http://7xritj.com1.z0.glb.clouddn.com/16-5-22/1408005.jpg)

**训练**：上面说到了网络训练的过程就是得到权值的过程，我们在开始训练之前网络的权值是随机初始化的，也就是我们的图片滤波器是随机初始化的。比如我们输入一张图片，随机初始化的CNN分类告诉我们有6%的可能是一个`网球场`，但实际上我们告诉CNN这是一个`飞机场`，然后其中会有一个[反向传播](https://zh.wikipedia.org/wiki/%E5%8F%8D%E5%90%91%E4%BC%A0%E6%92%AD%E7%AE%97%E6%B3%95)的处理过程来稍稍改变滤波器的参数以便它下次碰到相同的图片时会更可能的预测为`网球场`。然后我们对我们的训练数据重复这一过程，CNN里的滤波器会逐渐的调整到能够提取我们图片里的供我们分类的重要特征。

![](http://7xritj.com1.z0.glb.clouddn.com/16-7-8/73324267.jpg)


## 数据集分析

UC Merced Land Use数据集包含21类土地类型，每类图像为100张，每张图像的像素为256*256。
![](http://7xritj.com1.z0.glb.clouddn.com/16-5-22/46888988.jpg)
数据集特点，数据集比较小，每一类只有100张图片，这个数据集还有其他的一些特点比如类间距离小，如下图所示，不同类的图片之间很相似。
![](http://7xritj.com1.z0.glb.clouddn.com/16-5-23/95817534.jpg)

类内距离大，同类图片之间差别较大，如下图所示：
![](http://7xritj.com1.z0.glb.clouddn.com/16-5-23/31732082.jpg)

这些特点都是不利于图片的分类的，尤其是数据量太小，如果从头开始用数据集来训练网络肯定会造成严重的过拟合。考虑到这种情况，一个解决方法就是使用训练好的网络进行微调以适应我们自己的数据集，这种方法不仅能解决数据集小的问题，也能大大加快训练的速度。



## 网络微调

网络微调就是使用事先已经训练好的网络，对网络进行微小的改造再训练以适用与我们自己的数据库。为什么别人训练好的网络，我们自己拿到改改就能使用呢？就像之前所说的，CNN的卷积层是用来提取图像的特征的，事实上图片的线条一级色彩纹理大致上是一样的，也就是说一个训练好CNN网络的卷积层也可以用来提取其他数据集图像的特征，因为图像的特征基本相似。特别的，能够使用网络微调的一个重要因素是使用的事先训练好的网络使用的数据集要和我们自己的训练集图像之间的‘距离’要比较小。因为我们的数据集是光学遥感图像，所以和我们的光学图像在底层上的特征有非常强的相似性。

下图是Imagenet数据集的部分图片，也是我们要使用的预先训练好的所用网络的数据集。
![](http://7xritj.com1.z0.glb.clouddn.com/16-5-23/54435346.jpg)
基于遥感SAR图像每个像素级别的统计特性，这种用光学图像训练好的网络微调的方法是不适用与SAR图像分类的。SAR图像如下所示，直观上看也与光学图像差别很大。
![](http://7xritj.com1.z0.glb.clouddn.com/16-5-23/59029562.jpg)

我们选择Caffe里预先使用Imagnet训练好的CaffeNet网络来经行微调，CaffeNet网络结构如下所示，fc6前为CNN中的卷积层用来提取图像特征，f6、fc7、fc8为全连接层(可以看成是人工神经网络的输入层，隐含层，输出层)，因为CaffeNet网络是用来分类1000类的图像的，所以最后一层有1000个神经元。
![](http://7xritj.com1.z0.glb.clouddn.com/16-5-23/63655708.jpg)

而我们的数据集是分开21类的图像，所以微调网络中的调整主要就体现在这里，修改上述网络以使用我们自己的数据集，如下所示，只要把网络的输出层改为21个神经元即可。
![](http://7xritj.com1.z0.glb.clouddn.com/16-5-23/18035393.jpg)

我们说的要使用要使用预先训练好的网络就是要使用它事先训练好的权值，比较上述两个网络，只有最后一层不同，所以它们的其他层的权值的维数都是相同的，所以我们把CaffeNet训练好的权值直接用在我们自己定义的网络上，最后一层的权值则随机初始化并设置较大的学习速率，然后就可以用我们定义好的网络训练我们自己的数据集。

定义好网络之后就可以开始训练了，把数据集按4:1分为训练集和测试集，在测试集上的预测准确率在92%左右。

还有一种常用的方法是不用CNN的最后一层分类，用CNN提取到的特征用SVM来分类，也能达到不错的效果。在这里我们提取fc7层输出的特征，根据上面定义的网络结构，fc7层共有4096个神经元，所以每张图片的特征维数为4096维，维数比较大，所以我们使用SVM的线性核即可达到分类效果。

## 结果展示与分析

fine-turning结果展示：
其中对预测结果做了一些可视化展示，左图表示为预测前五类的概率，左右为图片真实的类别。
![](http://7xritj.com1.z0.glb.clouddn.com/16-5-23/60490439.jpg)
![](http://7xritj.com1.z0.glb.clouddn.com/16-5-23/91330851.jpg)
![](http://7xritj.com1.z0.glb.clouddn.com/16-5-23/20682353.jpg)

CNN提取特征，SVM分类结果展示：
![](http://7xritj.com1.z0.glb.clouddn.com/16-5-23/64622617.jpg)
![](http://7xritj.com1.z0.glb.clouddn.com/16-5-23/23497392.jpg)
![](http://7xritj.com1.z0.glb.clouddn.com/16-5-23/62208562.jpg)

从两种方法中可以看出，虽然都分类正确了，但用SVM作为分类器的正确分类的概率更高。

fine-turning方法每个类别的准确率：
![](http://7xritj.com1.z0.glb.clouddn.com/16-5-23/23347725.jpg)
从上图中我们可以看出，tenniscount类别的预测准确率最低，我们来看看有哪些tenniscount类是预测错了的：
![](http://7xritj.com1.z0.glb.clouddn.com/16-5-23/83947048.jpg)
![](http://7xritj.com1.z0.glb.clouddn.com/16-5-23/49629661.jpg)
从上面两个图片中可以看出，其实并不能说是预测错误，因为上面两张图中既包含了tenniscount类和CNN预测的类别，可以说本来就是有两个类。
### t-sne特征降维可视化

对CNN中第七层提取到的4096维特征经行降维可视化，从下图可以看出，分类准备率比较低的类别靠的都比较紧密，难以区分。
![](http://7xritj.com1.z0.glb.clouddn.com/16-7-8/84288255.jpg)

CNN+SVM每个类别的准确率：
![](http://7xritj.com1.z0.glb.clouddn.com/16-5-23/97608126.jpg)

## CNN中间层可视化

神经网络不仅仅是一个黑盒子，我们可以查看一些中间结果和参数。上面我们也说了一个卷积层就相当与一个图像滤波器，在上面的网络的第一层的卷积层中我们定义了96个滤波器，96个滤波器可视化如下图所示，学过图像处理的同学都知道，下图中第一个滤波器是提取斜向下的边缘特征，第二个滤波器是提取斜向上的边缘特征，前面的滤波器大多数是在提取边缘特征，后面的大多是在统计颜色特征。
![](http://7xritj.com1.z0.glb.clouddn.com/16-5-23/15829905.jpg)

我们输入一张图片，并输出其经过第一层卷积层滤波器滤波后的输出：
![](http://7xritj.com1.z0.glb.clouddn.com/16-5-23/12013586.jpg)
从第一层滤波后的结果可以看出，前面两个滤波器就是在显示斜向下和斜向上的边缘。

第五层卷积层滤波器输出如下图所示，高层的滤波器输出比较抽象。
![](http://7xritj.com1.z0.glb.clouddn.com/16-5-23/58707834.jpg)

## 总结及展望

当我们数据集不够的时候可以使用微调的方法，探索CNN怎么应用于SAR图像分类，解决图片类标签的分类问题。

## 代码地址
[land_use_CNN](https://github.com/BUPTLdy/Land_Use_CNN)

## 参考

[http://vision.ucmerced.edu/datasets/landuse.html](http://vision.ucmerced.edu/datasets/landuse.html)
[http://ufldl.stanford.edu/wiki/index.php/%E6%B1%A0%E5%8C%96](http://ufldl.stanford.edu/wiki/index.php/%E6%B1%A0%E5%8C%96)
[http://colah.github.io/posts/2014-07-Conv-Nets-Modular/](http://colah.github.io/posts/2014-07-Conv-Nets-Modular/)
[Tinker With a Neural Network Right Here in Your Browser](http://playground.tensorflow.org/#activation=tanh&batchSize=10&dataset=xor&regDataset=reg-plane&learningRate=0.03&regularizationRate=0&noise=0&networkShape=4&seed=0.38108&showTestData=false&discretize=false&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification)
