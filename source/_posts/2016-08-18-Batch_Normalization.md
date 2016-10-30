---
layout:     post
title:      "Implementation of Batch Normalization Layer"
subtitle:   "Batch Normalization实现"
date:       2016-08-18 10:00:00
author:     "Ldy"
header-img: "img/tag-bg1.jpg"
tags:
    - CNN
    - Deep Learning
---

![](http://7xritj.com1.z0.glb.clouddn.com/16-8-5/21841995.jpg)

<!--more-->

## 数据归一化
通常在神经网络训练开始前,都要对输入数据做一个归一化处理,那么具体为什么需要归一化呢?归一化后有什么好处呢?原因在于神经网络学习过程本质就是为了学习数据分布,一旦训练数据与测试数据的分布不同,那么网络的泛化能力也大大降低;另外一方面,一旦每批训练数据的分布各不相同(batch 梯度下降),那么网络就要在每次迭代都去学习适应不同的分布,这样将会大大降低网络的训练速度,这也正是为什么我们需要对数据都要做一个归一化预处理的原因。对于深度网络的训练是一个复杂的过程,只要网络的前面几层发生微小的改变,那么后面几层就会被累积放大下去。一旦网络某一层的输入数据的分布发生改变,那么这一层网络就需要去适应学习这个新的数据分布,所以如果训练过程中,训练数据的分布一直在发生变化,那么将会影响网络的训练速度。

<center>
![](http://i2.buimg.com/567571/70c7efd885eb08f3.png)
</center>

举例说明进行数据预处理能够加速训练过程，上图中红点代表2维的数据点，由于图像数据的每一维一般都是0-255之间的数字，因此数据点只会落在第一象限，而且图像数据具有很强的相关性，比如第一个灰度值为30，比较黑，那它旁边的一个像素值一般不会超过100，否则给人的感觉就像噪声一样。由于强相关性，数据点仅会落在第一象限的很小的区域中，形成类似上图所示的狭长分布。

而神经网络模型在初始化的时候，权重W是随机采样生成的，一个常见的神经元表示为：ReLU(Wx+b) = max(Wx+b,0)，即在Wx+b=0的两侧，对数据采用不同的操作方法。具体到ReLU就是一侧收缩，一侧保持不变。

随机的Wx+b=0表现为上图中的随机虚线，注意到，两条绿色虚线实际上并没有什么意义，在使用梯度下降时，可能需要很多次迭代才会使这些虚线对数据点进行有效的分割，就像紫色虚线那样，这势必会带来求解速率变慢的问题。更何况，我们这只是个二维的演示，数据占据四个象限中的一个，如果是几百、几千、上万维呢？而且数据在第一象限中也只是占了很小的一部分区域而已，可想而知不对数据进行预处理带来了多少运算资源的浪费，而且大量的数据外分割面在迭代时很可能会在刚进入数据中时就遇到了一个局部最优，导致overfit的问题。

这时，如果我们将数据减去其均值，数据点就不再只分布在第一象限，这时一个随机分界面落入数据分布的概率增加了多少呢？2^n倍！如果我们使用去除相关性的算法，例如PCA和ZCA白化，数据不再是一个狭长的分布，随机分界面有效的概率就又大大增加了。

不过计算协方差矩阵的特征值太耗时也太耗空间，我们一般最多只用到z-score处理，即每一维度减去自身均值，再除以自身标准差，这样能使数据点在每维上具有相似的宽度，可以起到一定的增大数据分布范围，进而使更多随机分界面有意义的作用。

## batch normalization 算法

算法基本流程：


<center>
![](http://7xritj.com1.z0.glb.clouddn.com/16-8-5/92943869.jpg)
</center>

如果在ReLU=max(Wx+b,0)之后，对数据进行归一化。然而，文章中说这样做在训练初期，分界面还在剧烈变化时，计算出的参数不稳定，所以退而求其次，在Wx+b之后进行归一化。因为初始的W是从标准高斯分布中采样得到的，而W中元素的数量远大于x，Wx+b每维的均值本身就接近0、方差接近1，所以在Wx+b后使用Batch Normalization能得到更稳定的结果。

文中使用了类似z-score的归一化方式：每一维度减去自身均值，再除以自身标准差，由于使用的是随机梯度下降法，这些均值和方差也只能在当前迭代的batch中计算，故作者给这个算法命名为Batch Normalization。

在Normalization完成后，Google的研究员仍对数值稳定性不放心，又加入了两个参数gamma和beta，使得

$$y\_i=\gamma \hat{x}\_i+ \beta$$

注意到，如果我们令gamma等于之前求得的标准差，beta等于之前求得的均值，则这个变换就又将数据还原回去了。在他们的模型中，这两个参数与每层的W和b一样，是需要迭代求解的。为什么进行归一化之后又添加两个可学习的参数对数据进行变化：**实际上BN可以看作是在原模型上加入的“新操作”，这个新操作很大可能会改变某层原来的输入。当然也可能不改变，不改变的时候就是“还原原来输入”。如此一来，既可以改变同时也可以保持原输入，那么模型的容纳能力（capacity）就提升了。**


## 算法实现

![](http://7xritj.com1.z0.glb.clouddn.com/16-8-5/21841995.jpg)

根据链式求导法则，我们可以把复杂的运算分解成一步一步能够简单求导的运算，然后根据链式求导法则来求得最终的导数，参考[cs231n](http://cs231n.github.io/optimization-2/)。

### 前向传播
```python
def batchnorm_forward(x, gamma, beta, eps):

  N, D = x.shape

  #step1: calculate mean
  mu = 1./N * np.sum(x, axis = 0)

  #step2: subtract mean vector of every trainings example
  xmu = x - mu

  #step3: following the lower branch - calculation denominator
  sq = xmu ** 2

  #step4: calculate variance
  var = 1./N * np.sum(sq, axis = 0)

  #step5: add eps for numerical stability, then sqrt
  sqrtvar = np.sqrt(var + eps)

  #step6: invert sqrtwar
  ivar = 1./sqrtvar

  #step7: execute normalization
  xhat = xmu * ivar

  #step8: Nor the two transformation steps
  gammax = gamma * xhat

  #step9
  out = gammax + beta

  #store intermediate
  cache = (xhat,gamma,xmu,ivar,sqrtvar,var,eps)

  return out, cache
```

### 反向求导
```python
def batchnorm_backward(dout, cache):

  #unfold the variables stored in cache
  xhat,gamma,xmu,ivar,sqrtvar,var,eps = cache

  #get the dimensions of the input/output
  N,D = dout.shape

  #step9
  dbeta = np.sum(dout, axis=0)
  dgammax = dout #not necessary, but more understandable

  #step8
  dgamma = np.sum(dgammax*xhat, axis=0)
  dxhat = dgammax * gamma

  #step7
  divar = np.sum(dxhat*xmu, axis=0)
  dxmu1 = dxhat * ivar

  #step6
  dsqrtvar = -1. /(sqrtvar**2) * divar

  #step5
  dvar = 0.5 * 1. /np.sqrt(var+eps) * dsqrtvar

  #step4
  dsq = 1. /N * np.ones((N,D)) * dvar

  #step3
  dxmu2 = 2 * xmu * dsq

  #step2
  dx1 = (dxmu1 + dxmu2)
  dmu = -1 * np.sum(dxmu1+dxmu2, axis=0)

  #step1
  dx2 = 1. /N * np.ones((N,D)) * dmu

  #step0
  dx = dx1 + dx2

  return dx, dgamma, dbeta
```

## 卷积层batch normalization

这里有一点需要注意，像卷积层这样具有权值共享的层，Wx+b的均值和方差是对整张map求得的，在batch_size \* channel \* height \* width这么大的一层中，对总共batch_size\*height\*width个像素点统计得到一个均值和一个标准差，共得到channel组参数。

也就是说把每个channel看出一批数据，然后就可以调用全连接层的batch normalization 算法了。

```python
def spatial_batchnorm_forward(x, gamma, beta, bn_param):
  """
  Computes the forward pass for spatial batch normalization.

  Inputs:
  - x: Input data of shape (N, C, H, W)
  - gamma: Scale parameter, of shape (C,)
  - beta: Shift parameter, of shape (C,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance. momentum=0 means that
      old information is discarded completely at every time step, while
      momentum=1 means that new information is never incorporated. The
      default of momentum=0.9 should work well in most situations.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features

  Returns a tuple of:
  - out: Output data, of shape (N, C, H, W)
  - cache: Values needed for the backward pass
  """
  out, cache = None, None
  N, C, H, W = x.shape
  x_flat = x.transpose(0, 2, 3, 1).reshape(-1, C)
  out_flat, cache = batchnorm_forward(x_flat, gamma, beta, bn_param)
  out = out_flat.reshape(N, H, W, C).transpose(0, 3, 1, 2)

  return out, cache


def spatial_batchnorm_backward(dout, cache):
  """
  Computes the backward pass for spatial batch normalization.

  Inputs:
  - dout: Upstream derivatives, of shape (N, C, H, W)
  - cache: Values from the forward pass

  Returns a tuple of:
  - dx: Gradient with respect to inputs, of shape (N, C, H, W)
  - dgamma: Gradient with respect to scale parameter, of shape (C,)
  - dbeta: Gradient with respect to shift parameter, of shape (C,)
  """
  dx, dgamma, dbeta = None, None, None                                   
  N, C, H, W = dout.shape
  dout_flat = dout.transpose(0, 2, 3, 1).reshape(-1, C)
  dx_flat, dgamma, dbeta = batchnorm_backward(dout_flat, cache)
  dx = dx_flat.reshape(N, H, W, C).transpose(0, 3, 1, 2)                        
  return dx, dgamma, dbeta
```

## 参考

[Batch Normalization 学习笔记](http://blog.csdn.net/hjimce/article/details/50866313)

[《Batch Normalization Accelerating Deep Network Training by Reducing Internal Covariate Shift》阅读笔记与实现](http://blog.csdn.net/happynear/article/details/44238541)

[深度学习中 Batch Normalization为什么效果好？ - 回答作者: 魏秀参](http://zhihu.com/question/38102762/answer/85238569)

[Understanding the backward pass through Batch Normalization Layer](https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html)
