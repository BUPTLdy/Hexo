---
layout:     post
title:      "Keras Introduction"
subtitle:   "Keras简介"
date:       2016-04-07 11:00:00
author:     "Ldy"
header-img: "img/tag-bg1.jpg"
tags:
    - Python
    - Keras
    - Deep Learning
---

<center>
![](http://7xritj.com1.z0.glb.clouddn.com/16-4-7/16129390.jpg)
</center>

# Keras 简介

[Keras](http://keras.io/)是一个用Python编写的基于 TensorFlow 和 Theano高度模块化的神经网络库。其最大的优点在于样例丰富，现有主流模型封装完美。复杂点的模型可以像搭积木一样搞出来，适合快速地搭建模型。

<!--more-->
安装：
```python
sudo pip install keras
```

# Keras里的基本模块
## optimizers
Keras包含了很多优化方法。比如最常用的随机梯度下降法(SGD)，还有Adagrad、Adadelta、RMSprop、Adam等。下面通过具体的代码介绍一下优化器的使用方法。
在编译一个Keras模型时，优化器是2个参数之一（另外一个是损失函数）。看如下代码：

```python
model = Sequential()  
model.add(Dense(64, init='uniform', input_dim=10))  
model.add(Activation('tanh'))  
model.add(Activation('softmax'))  

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)  
model.compile(loss='mean_squared_error', optimizer=sgd)  
```

这个例子中是在调用compile之前实例化了一个优化器。我们也可以通过传递名字的方式调用默认的优化器。代码如下：

```python
# passoptimizer by name: default parameters will be used  
model.compile(loss='mean_squared_error', optimizer='sgd')
```

SGD（随机梯度下降优化器，性价比最好的算法）

```python
keras.optimizers.SGD(lr=0.01, momentum=0., decay=0., nesterov=False)  
```

参数：

- lr :float>=0，学习速率
- momentum :float>=0 参数更新的动量
- decay : float>=0 每次更新后学习速率的衰减量
- nesterov :Boolean 是否使用Nesterov动量项

## objectives

目标函数模块，keras提供了mean_squared_error，mean_absolute_error，squared_hinge，hinge，binary_crossentropy，categorical_crossentropy这几种目标函数。

这里binary_crossentropy 和categorical_crossentropy也就是常说的logloss.

## Activations

激活函数模块，keras提供了linear、sigmoid、hard_sigmoid、tanh、softplus、relu、softplus，另外softmax也放在Activations模块里。此外，像LeakyReLU和PReLU这种比较新的激活函数，keras在keras.layers.advanced_activations模块里提供。

## initializations

权值初始化，在Keras中对权值矩阵初始化的方式很简单，就是在add某一层时，同时注明初始化该层的概率分布是什么就可以了。代码如下：

```python
# init是关键字，’uniform’表示用均匀分布去初始化  
model.add(Dense(64, init='uniform'))  
```
keras提供了uniform、lecun_uniform、normal、orthogonal、zero、glorot_normal、he_normal这几种。

## regularizers

深度学习容易出现过拟合，通过使用[正则化方法](http://blog.csdn.net/u012162613/article/details/44261657)，防止过拟合，提高泛化能力。

使用示例代码如下：

```python
from keras.regularizers import l2, activity_l2  
model.add(Dense(64, input_dim=64, W_regularizer=l2(0.01), activity_regularizer=activity_l2(0.01)))
```

## constraints

除了正则化外，Keras还有一个约束限制功能。函数可以设置在训练网络到最优时对网络参数的约束。这个约束就是限制参数值的取值范围。比如最大值是多少，不允许为负值等。

2个关键的参数：

- W_constraint：约束主要的权值矩阵
- b_constraint：约束偏置值

使用示例代码如下：
```python
from keras.constraints import maxnorm
model.add(Dense(64, W_constraint =maxnorm(2)))
#限制权值的各个参数不能大于2
```
可用的约束限制

- maxnorm(m=2): 最大值约束
- nonneg(): 不允许负值
- unitnorm(): 归一化

# 实例：解决XOR问题

```python
import numpy as np
from keras.models import Sequential
from keras.layers.core import Activation, Dense
from keras.optimizers import SGD

X = np.zeros((4, 2), dtype='uint8')#训练数据
y = np.zeros(4, dtype='uint8')#训练标签

X[0] = [0, 0]
y[0] = 0
X[1] = [0, 1]
y[1] = 1
X[2] = [1, 0]
y[2] = 1
X[3] = [1, 1]
y[3] = 0

model = Sequential()#实例化模型
model.add(Dense(2, input_dim=2))#输入层，输入数据维数为2
model.add(Activation('sigmoid'))#设置激活函数
model.add(Dense(1))
model.add(Activation('sigmoid'))

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer=sgd)

history = model.fit(X, y, nb_epoch=10000, batch_size=4, show_accuracy=True, verbose=2)

print model.predict(X)#预测

```

# 参考

[Keras Documentation](http://keras.io/)

[Keras 学习随笔](http://www.lai18.com/user/301164.html)

[深度学习框架Keras简介](http://blog.csdn.net/u012162613/article/details/45397033)
