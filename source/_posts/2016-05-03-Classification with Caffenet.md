---
layout:     post
title:      "Classification with Caffenet"
subtitle:   "使用Caffe训练好的CaffeNet进行图片分类"
date:       2016-05-03 10:00:00
author:     "Ldy"
header-img: "img/tag-bg1.jpg"
tags:
    - Caffe
    - Deep Learning
---
<center>
![](http://7xritj.com1.z0.glb.clouddn.com/16-5-3/46730593.jpg)
</center>

<!--more-->

Caffe直接使用训练好的CaffeNet模型来进行分类，Caffe的安装有很多教程，[千秋轻松装Caffe教程（含CUDA 7.0和CuDNN）
](http://ixez.info/?p=105)这个教程说的很详细，其中比较繁琐的就是CUDA的安装了，可以参考这里：[Deepin CUDA安装及Keras使用GPU模式运行](http://buptldy.github.io/2016/04/09/2016-04-09-Deepin%20CUDA%E5%AE%89%E8%A3%85%E5%8F%8AKeras%E4%BD%BF%E7%94%A8GPU%E6%A8%A1%E5%BC%8F%E8%BF%90%E8%A1%8C/)。其中遇到的一个比较大的坑就是cuDNN的安装，首先得确定你的GPU是否支持cuDNN，cuDNN要求GPU的计算能力在3.0以上，这里[ http://developer.nvidia.com/cuda-gpus]( http://developer.nvidia.com/cuda-gpus)可以查询GPU的计算能力，也能查询你的GPU是否支持CUDA，如果你的GPU不支持cuDNN但是支持CUDA，在编译配置文件注释掉`USE_CUDNN :=1`和`CPU_ONLY :=1`就可以使用CUDA了。如果你的GPU支持GUDA和cuDNN，得注意你下的Caffe所支持cuDNN的版本，这里可以查看[http://caffe.berkeleyvision.org/installation.html](http://caffe.berkeleyvision.org/installation.html)。

在这里我们比较下CPU和GPU模式下，网络的运行速度，并了解模型特征的提取。

## 设置环境

导入Python,numpy,matplotlib

```python
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
# set display defaults
plt.rcParams['figure.figsize'] = (10, 10)        # large images
plt.rcParams['image.interpolation'] = 'nearest'  # don't interpolate: show square pixels
plt.rcParams['image.cmap'] = 'gray'  # use grayscale output rather than a (potentially misleading) color heatmap
```

导入caffe，其中注意caffe的路径设置

```python
import sys
caffe_root='/home/ldy/workspace/caffe/' #设置你caffe的安装目录
sys.path.insert(0,caffe_root+'python')
import caffe                            #导入caffe
```

第一次运行需要联网下载模型

```python
import os
if os.path.isfile(caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'):
    print 'CaffeNet found.'
else:
    print 'Downloading pre-trained CaffeNet model...'
    !/home/ldy/workspace/caffe/scripts/download_model_binary.py /home/ldy/workspace/caffe/models/bvlc_reference_caffenet
```

    CaffeNet found.


## 设置网络并对输入进行处理

设置CPU模式并从本地加载网络

```python
caffe.set_mode_cpu()

model_def = caffe_root + 'models/bvlc_reference_caffenet/deploy.prototxt'
model_weights = caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'

net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

```

设置输入处理

CaffeNet默认的输入图像格式是BGR模式，像素值是[0,255]然后减去ImageNet的像素平均值，而且图像通道的维数是在第一维。

matplotlib导入图像的格式是RGB,像素值的范围是[0,1]，通道维数在第三维，所以我们需要进行转换。
```python
# load the mean ImageNet image (as distributed with Caffe) for subtraction
mu = np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
print 'mean-subtracted values:', zip('BGR', mu)

# create transformer for the input called 'data'
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR
```

    mean-subtracted values: [('B', 104.0069879317889), ('G', 116.66876761696767), ('R', 122.6789143406786)]


## CPU模式分类

设置输入的大小

```python
# set the size of the input (we can skip this if we're happy
#  with the default; we can also change it later, e.g., for different batch sizes)
net.blobs['data'].reshape(50,        # batch size
                          3,         # 3-channel (BGR) images
                          227, 227)  # image size is 227x227
```

加载图片并转换

```python
image = caffe.io.load_image(caffe_root + 'examples/images/cat.jpg')
transformed_image = transformer.preprocess('data', image)
plt.imshow(image)
```
    <matplotlib.image.AxesImage at 0x7f7ba44f0a50>

<center>
![](http://7xritj.com1.z0.glb.clouddn.com/16-5-3/52956146.jpg)
</center>

进行分类

```python
# copy the image data into the memory allocated for the net
net.blobs['data'].data[...] = transformed_image

### perform classification
output = net.forward()

output_prob = output['prob'][0]  # the output probability vector for the first image in the batch

print 'predicted class is:', output_prob.argmax()
```

    predicted class is: 281


从上面的输出，我们得到输入的图片得到的类别可能是第281类，但是并不知道它对应的标签，下面我们来加载ImageNet的标签(首次需要联网)。

```python
# load ImageNet labels
labels_file = caffe_root + 'data/ilsvrc12/synset_words.txt'
if not os.path.exists(labels_file):
    !/home/ldy/workspace/caffe/data/ilsvrc12/get_ilsvrc_aux.sh

labels = np.loadtxt(labels_file, str, delimiter='\t')

print 'output label:', labels[output_prob.argmax()]
```

    Downloading...
    --2016-05-03 10:54:43--  http://dl.caffe.berkeleyvision.org/caffe_ilsvrc12.tar.gz
    正在解析主机 dl.caffe.berkeleyvision.org (dl.caffe.berkeleyvision.org)... 169.229.222.251
    正在连接 dl.caffe.berkeleyvision.org (dl.caffe.berkeleyvision.org)|169.229.222.251|:80... 已连接。
    已发出 HTTP 请求，正在等待回应... 200 OK
    长度：17858008 (17M) [application/octet-stream]
    正在保存至: “caffe_ilsvrc12.tar.gz”

    caffe_ilsvrc12.tar. 100%[===================>]  17.03M  2.54MB/s    in 8.9s    

    2016-05-03 10:54:53 (1.91 MB/s) - 已保存 “caffe_ilsvrc12.tar.gz” [17858008/17858008])

    Unzipping...
    Done.
    output label: n02123045 tabby, tabby cat

现在我们得到了输出为｀tabby cat｀，如果我们想得到其他的可能类别，如下所示：

```python
# sort top five predictions from softmax output
top_inds = output_prob.argsort()[::-1][:5]  # reverse sort and take five largest items

print 'probabilities and labels:'
zip(output_prob[top_inds], labels[top_inds])
```

    probabilities and labels:
    [(0.31243625, 'n02123045 tabby, tabby cat'),
     (0.23797135, 'n02123159 tiger cat'),
     (0.12387258, 'n02124075 Egyptian cat'),
     (0.10075716, 'n02119022 red fox, Vulpes vulpes'),
     (0.070957333, 'n02127052 lynx, catamount')]


## 切换到GPU模式


查看CPU模式花费的时间

```python
%timeit net.forward()
```

    1 loop, best of 3: 8.87 s per loop


切换到GPU模式，查看花费时间

```python
#caffe.set_device(0)  # if we have multiple GPUs, pick the first one
caffe.set_mode_gpu()
net.forward()  # run once before timing to set up memory
%timeit net.forward()
```

    1 loop, best of 3: 2.27 s per loop


##查看中间输入

神经网络不仅仅是一个黑盒子，我们可以查看一些中间结果和参数。

查看激活函数输出的数据维数，格式为(batch_size, channel_dim, height, width)。

```python
# for each layer, show the output shape
for layer_name, blob in net.blobs.iteritems():
    print layer_name + '\t' + str(blob.data.shape)
```

    data	(50, 3, 227, 227)
    conv1	(50, 96, 55, 55)
    pool1	(50, 96, 27, 27)
    norm1	(50, 96, 27, 27)
    conv2	(50, 256, 27, 27)
    pool2	(50, 256, 13, 13)
    norm2	(50, 256, 13, 13)
    conv3	(50, 384, 13, 13)
    conv4	(50, 384, 13, 13)
    conv5	(50, 256, 13, 13)
    pool5	(50, 256, 6, 6)
    fc6	(50, 4096)
    fc7	(50, 4096)
    fc8	(50, 1000)
    prob	(50, 1000)

查看权值参数的维数，权值格式为(output_channels, input_channels, filter_height, filter_width)，偏置的格式为(output_channels,)。

```python
for layer_name, param in net.params.iteritems():
    print layer_name + '\t' + str(param[0].data.shape), str(param[1].data.shape)

```

    conv1	(96, 3, 11, 11) (96,)
    conv2	(256, 48, 5, 5) (256,)
    conv3	(384, 256, 3, 3) (384,)
    conv4	(384, 192, 3, 3) (384,)
    conv5	(256, 192, 3, 3) (256,)
    fc6	(4096, 9216) (4096,)
    fc7	(4096, 4096) (4096,)
    fc8	(1000, 4096) (1000,)


## 输出可视化

```python
def vis_square(data):
    """Take an array of shape (n, height, width) or (n, height, width, 3)
       and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)"""

    # normalize data for display
    data = (data - data.min()) / (data.max() - data.min())

    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = (((0, n ** 2 - data.shape[0]),
               (0, 1), (0, 1))                 # add some space between filters
               + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)
    data = np.pad(data, padding, mode='constant', constant_values=1)  # pad with ones (white)

    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])

    plt.imshow(data); plt.axis('off')
```

第一层卷积滤波器
```python
# the parameters are a list of [weights, biases]
filters = net.params['conv1'][0].data
vis_square(filters.transpose(0, 2, 3, 1))
```

<center>
![](http://7xritj.com1.z0.glb.clouddn.com/16-5-3/10647296.jpg)
</center>

第一层卷积层的输出
```python
feat = net.blobs['conv1'].data[0, :36]
vis_square(feat)
```

<center>
![](http://7xritj.com1.z0.glb.clouddn.com/16-5-3/25648526.jpg)
</center>

第五层pooling之后的输出
```python
feat = net.blobs['pool5'].data[0]
vis_square(feat)
```

<center>
![](http://7xritj.com1.z0.glb.clouddn.com/16-5-3/47782240.jpg)
</center>

第一个全连接层的输出

```python
feat = net.blobs['fc6'].data[0]
plt.subplot(2, 1, 1)
plt.plot(feat.flat)
plt.subplot(2, 1, 2)
_ = plt.hist(feat.flat[feat.flat > 0], bins=100)
```

<center>
![](http://7xritj.com1.z0.glb.clouddn.com/16-5-3/71842859.jpg)
</center>

最后的类别概率输出
```python
feat = net.blobs['prob'].data[0]
plt.figure(figsize=(15, 3))
plt.plot(feat.flat)
```
    [<matplotlib.lines.Line2D at 0x7f7ba0177d10>]

<center>
![](http://7xritj.com1.z0.glb.clouddn.com/16-5-3/77194220.jpg)
</center>

## 对自己的图片分类

设置好图片的链接就好了

```python
# download an image
#my_image_url = "..."  # paste your URL here
# for example:
my_image_url = "https://upload.wikimedia.org/wikipedia/commons/b/be/Orang_Utan%2C_Semenggok_Forest_Reserve%2C_Sarawak%2C_Borneo%2C_Malaysia.JPG"
!wget -O image.jpg $my_image_url

# transform it and copy it into the net
image = caffe.io.load_image('image.jpg')
net.blobs['data'].data[...] = transformer.preprocess('data', image)

# perform classification
net.forward()

# obtain the output probabilities
output_prob = net.blobs['prob'].data[0]

# sort top five predictions from softmax output
top_inds = output_prob.argsort()[::-1][:5]

plt.imshow(image)

print 'probabilities and labels:'
zip(output_prob[top_inds], labels[top_inds])
```

    --2016-05-03 11:23:33--  https://upload.wikimedia.org/wikipedia/commons/b/be/Orang_Utan%2C_Semenggok_Forest_Reserve%2C_Sarawak%2C_Borneo%2C_Malaysia.JPG
    正在解析主机 upload.wikimedia.org (upload.wikimedia.org)... 2620:0:863:ed1a::2:b, 2620:0:863:ed1a::2:b, 198.35.26.112, ...
    正在连接 upload.wikimedia.org (upload.wikimedia.org)|2620:0:863:ed1a::2:b|:443... 已连接。
    已发出 HTTP 请求，正在等待回应... 200 OK
    长度：1443340 (1.4M) [image/jpeg]
    正在保存至: “image.jpg”

    image.jpg           100%[===================>]   1.38M  1.41MB/s    in 1.0s    

    2016-05-03 11:23:35 (1.41 MB/s) - 已保存 “image.jpg” [1443340/1443340])

    probabilities and labels:

    [(0.9680779, 'n02480495 orangutan, orang, orangutang, Pongo pygmaeus'),
     (0.030589299, 'n02492660 howler monkey, howler'),
     (0.00085892546, 'n02493509 titi, titi monkey'),
     (0.00015429084, 'n02493793 spider monkey, Ateles geoffroyi'),
     (7.2596376e-05, 'n02488291 langur')]


<center>
![](http://7xritj.com1.z0.glb.clouddn.com/16-5-3/25241651.jpg)
</center>

## 参考
[Classification: Instant Recognition with Caffe](http://nbviewer.jupyter.org/github/BVLC/caffe/blob/master/examples/00-classification.ipynb)
