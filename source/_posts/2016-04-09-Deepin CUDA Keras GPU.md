---
layout:     post
title:      "Deepin CUDA Install and Run Keras on GPU"
subtitle:   "Deepin CUDA安装及Keras使用GPU模式运行"
date:       2016-04-09 11:00:00
author:     "Ldy"
header-img: "img/tag-bg1.jpg"
tags:
    - Cuda
    - Keras
    - Deep Learning
---

<center>
![](http://7xritj.com1.z0.glb.clouddn.com/16-4-9/8883646.jpg)
</center>

<!--more-->
# Deepin简介
[Deepin](https://www.deepin.org/)是由武汉深之度科技有限公司开发的Linux发行版,Deepin 为所有人提供稳定、高效的操作系统，强调安全、易用、美观。其口号为“免除新手痛苦，节约老手时间”。

# cuda安装

## 下载

按照系统的版本下载对应的cuda版本，下载地址：[https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)


## 安装

注意执行安装文件的时候一定要加上'--­­override'，不然会出现错误：'"Toolkit: Installation Failed. Using unsupported Compiler."'
```shell
chmod 755 cuda_7.5.18_linux.run
sudo ./cuda_7.5.18_linux.run --­­override
```

**如果你电脑里已经装好比cuda内置的NVIDIA驱动更新的版本，那么在安装的时候就不要选择安装NVIDIA驱动。**

安装过程的设置如下所示：
```shell
-------------------------------------------------------------
Do you accept the previously read EULA? (accept/decline/quit): accept
You are attempting to install on an unsupported configuration. Do you wish to continue? ((y)es/(n)o) [ default is no ]: y
Install NVIDIA Accelerated Graphics Driver for Linux-x86_64 352.39? ((y)es/(n)o/(q)uit): n
Install the CUDA 7.5 Toolkit? ((y)es/(n)o/(q)uit): y
Enter Toolkit Location [ default is /usr/local/cuda-7.5 ]:
Do you want to install a symbolic link at /usr/local/cuda? ((y)es/(n)o/(q)uit): y
Install the CUDA 7.5 Samples? ((y)es/(n)o/(q)uit): y
Enter CUDA Samples Location [ default is /home/kinghorn ]: /usr/local/cuda-7.5
Installing the CUDA Toolkit in /usr/local/cuda-7.5 ...
Finished copying samples.

===========
= Summary =
===========

Driver:   Not Selected
Toolkit:  Installed in /usr/local/cuda-7.5
Samples:  Installed in /usr/local/cuda-7.5
```

## 环境设置

打开~/.bashrc
```
gedit ~/.bashrc
```
添加下面两条语句：
```

export PATH=$PATH:/usr/local/cuda/bin

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib
```


## 强制cuda使用gcc 5

因为cuda默认不使用gcc>4.8，通过注释掉报错行来强制使用gcc 5。
```
sudo gedit /usr/local/cuda/include/host_config.h

//注释掉115行
//#error -- unsupported GNU version! gcc versions later than 4.9 are not supported!
```

## 运行cuda内置的例子

为了测试是否安装成功

进入内置例程
```
 cd /usr/local/cuda/samples/1_Utilities/deviceQuery
```

编译
```
make
```

运行
```
 ./deviceQuery
```
得到结果：
```
CUDA Device Query (Runtime API) version (CUDART static linking)

Detected 1 CUDA Capable device(s)

Device 0: "GeForce GT 520M"
 CUDA Driver Version / Runtime Version          8.0 / 7.5
 CUDA Capability Major/Minor version number:    2.1
 Total amount of global memory:                 1024 MBytes (1073414144 bytes)
 ( 1) Multiprocessors, ( 48) CUDA Cores/MP:     48 CUDA Cores
 GPU Max Clock rate:                            1480 MHz (1.48 GHz)
 Memory Clock rate:                             800 Mhz
 Memory Bus Width:                              64-bit
 L2 Cache Size:                                 65536 bytes
 Maximum Texture Dimension Size (x,y,z)         1D=(65536), 2D=(65536, 65535), 3D=(2048, 2048, 2048)
 Maximum Layered 1D Texture Size, (num) layers  1D=(16384), 2048 layers
 Maximum Layered 2D Texture Size, (num) layers  2D=(16384, 16384), 2048 layers
 Total amount of constant memory:               65536 bytes
 Total amount of shared memory per block:       49152 bytes
 Total number of registers available per block: 32768
 Warp size:                                     32
 Maximum number of threads per multiprocessor:  1536
 Maximum number of threads per block:           1024
 Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
 Max dimension size of a grid size    (x,y,z): (65535, 65535, 65535)
 Maximum memory pitch:                          2147483647 bytes
 Texture alignment:                             512 bytes
 Concurrent copy and kernel execution:          Yes with 1 copy engine(s)
 Run time limit on kernels:                     No
 Integrated GPU sharing Host Memory:            No
 Support host page-locked memory mapping:       Yes
 Alignment requirement for Surfaces:            Yes
 Device has ECC support:                        Disabled
 Device supports Unified Addressing (UVA):      Yes
 Device PCI Domain ID / Bus ID / location ID:   0 / 1 / 0
 Compute Mode:
    < Default (multiple host threads can use ::cudaSetDevice() with device simultaneously) >

deviceQuery, CUDA Driver = CUDART, CUDA Driver Version = 8.0, CUDA Runtime Version = 7.5, NumDevs = 1, Device0 = GeForce GT 520M
Result = PASS

```

如果编译出错，检查是否有强制设置gcc 5来编译；如果输出结果为fail，说明没有检查到显卡，解决方案是升级你的NVIDIA驱动，确保你电脑的NVIDIA驱动版本要不低于cuda的内置版本。

# 设置Keras运行于GPU模式


## 方法一
 使用如下命令行运行
```python
THEANO_FLAGS=device=gpu,floatX=float32 python my_keras_script.py
```
## 方法二

设置$HOME/.theanorc文件

添加如下所示文件
```
[global]
floatX = float32
device = gpu

[lib]
cnmem = 0.9

[cuda]
root = /usr/local/cuda
```
## 方法三

在你的代码前面，加上如下所示代码：
```python
import theano
theano.config.device = 'gpu'
theano.config.floatX = 'float32'
```

我们来运行Keras里的一个用于电影评论情感分析的例子[imdb_cnn.py](https://github.com/fchollet/keras/blob/master/examples/imdb_cnn.py),第一次运行时需要联网，要下载数据库。
```python
'''This example demonstrates the use of Convolution1D for text classification.
Run on GPU: THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python imdb_cnn.py
Get to 0.835 test accuracy after 2 epochs. 100s/epoch on K520 GPU.
'''

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.datasets import imdb


# set parameters:
max_features = 5000
maxlen = 100
batch_size = 32
embedding_dims = 100
nb_filter = 250
filter_length = 3
hidden_dims = 250
nb_epoch = 2

print('Loading data...')
(X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=max_features,
                                                      test_split=0.2)
print(len(X_train), 'train sequences')
print(len(X_test), 'test sequences')

print('Pad sequences (samples x time)')
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

print('Build model...')
model = Sequential()

# we start off with an efficient embedding layer which maps
# our vocab indices into embedding_dims dimensions
model.add(Embedding(max_features, embedding_dims, input_length=maxlen))
model.add(Dropout(0.25))

# we add a Convolution1D, which will learn nb_filter
# word group filters of size filter_length:
model.add(Convolution1D(nb_filter=nb_filter,
                        filter_length=filter_length,
                        border_mode='valid',
                        activation='relu',
                        subsample_length=1))
# we use standard max pooling (halving the output of the previous layer):
model.add(MaxPooling1D(pool_length=2))

# We flatten the output of the conv layer,
# so that we can add a vanilla dense layer:
model.add(Flatten())

# We add a vanilla hidden layer:
model.add(Dense(hidden_dims))
model.add(Dropout(0.25))
model.add(Activation('relu'))

# We project onto a single unit output layer, and squash it with a sigmoid:
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop')
model.fit(X_train, y_train, batch_size=batch_size,
          nb_epoch=nb_epoch, show_accuracy=True,
          validation_data=(X_test, y_test))
```


运行这个例子，在K520 GPU上是100s一次循环，我电脑显卡型号为GeForce GT 520M，大概需要175s一次循环，不过比在cpu上运行快多啦，在我这四年前旧电脑cpu上运行差不多要一个小时。

# 参考
[	NVIDIA CUDA with Ubuntu 16.04 beta on a laptop](https://www.pugetsystems.com/labs/articles/NVIDIA-CUDA-with-Ubuntu-16-04-beta-on-a-laptop-if-you-just-cannot-wait-775/)

[Keras FAQ](http://keras.io/faq/#how-can-i-run-keras-on-gpu)
