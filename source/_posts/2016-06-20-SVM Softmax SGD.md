---
layout:     post
title:      "Solve Linear Classifier by SGD"
subtitle:   "线性SVM及Softmax的随机梯度下降求解"
date:       2016-06-20 10:00:00
author:     "Ldy"
header-img: "img/tag-bg1.jpg"
tags:
    - SVM
    - Softmax
---
![](http://7xritj.com1.z0.glb.clouddn.com/16-6-20/61070235.jpg)
<!--more-->

## 线性分类器
一个线性分类器的基本形式如下所示：
$$f(x_i,W,b)=Wx_i+b （1）$$
在上面的公式中，如果是对图像经行分类，$x_i$表示对一张图片展开成一个列向量维数为[D,1],矩阵**W**维数为[K,D],向量**b**维数为[K,1]。参数**W**通常成为权值，**b**为偏置。


### 导入数据

```python
import numpy as np
from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt

# This is a bit of magic to make matplotlib figures appear inline in the
# notebook rather than in a new window.
%matplotlib inline
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

digits=load_digits()
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1)
print X_train.shape, y_train.shape
print X_val.shape, y_val.shape
print X_test.shape, y_test.shape
```

    (1293, 64) (1293,)
    (144, 64) (144,)
    (360, 64) (360,)



```python
classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
num_classes = len(classes)
samples_per_class = 3
for y, cls in enumerate(classes):
    idxs = np.flatnonzero(y_train == y)
    idxs = np.random.choice(idxs, samples_per_class, replace=False)
    for i, idx in enumerate(idxs):
        plt_idx = i * num_classes + y + 1
        plt.subplot(samples_per_class, num_classes, plt_idx)
        plt.imshow(X_train[idx].astype('uint8').reshape(8,8))
        plt.axis('off')
        if i == 0:
            plt.title(cls)
plt.show()
```


![](http://7xritj.com1.z0.glb.clouddn.com/16-6-20/14165529.jpg)


比如在数字手写识别数据集中，要分类[0-9]共十类数字手写图片，每张图片的像素为8×8，如上图所示。分类器的目的就是通过训练得到参数**W,b**，应为我们知道输入数据是 $(x_i,y_i)$ （$x_i$是输入图片像素值，$y_i$为对应类别号）是给定而且是固定的，我们的目标就是通过控制参数**W,b**来尽量拟合公式(1), 使得公式(1)能通过参数对输入数据$x_i$计算得到正确的$y_i$。

**Bias trick**,在公式(1)中有两个参数**W,b**，通过一个小技巧可以把这两个参数组合在一个矩阵中，通过把$x_i$增加一维，设置值为1，就可以把公式(1)写为：
$$f(x_i,W)=Wx_i$$
原理如下图所示：
![](http://7xritj.com1.z0.glb.clouddn.com/16-6-20/17970270.jpg)


```python
X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])

print X_train.shape, X_val.shape, X_test.shape
```

    (1293, 65) (144, 65) (360, 65)


## 损失函数

上面说到使用公式$f(x_i,W)=Wx_i$对输入图片经行每一类的打分，但是开始时线性分类器预测的打分和我们真实的类别可能相差比较远，我们需要一个函数来表示真实的分数和分类器所计算到的分数之间的距离，这个函数就叫损失函数。

比如说我们输入一张图像像素值$x_i$，其真实类别为$y_i$,我们通过分类器计算每类的得分 $f(x_i,W)$ 。 例如 $s\_j=f(x\_i,W)\_ j$ 表示分类器对输入数据 $x\_i$ 预测为第j类的可能性，那么损失函数就可以定义为：

$$L\_i=\sum \_{j\neq y\_i}max(0,s\_j-s\_{y\_i}+\Delta) (2)$$

假如我们有三类通过分类器得到每类的分数为[13,-7,11],并假设第一类是正确的类别($y\_i=0$), 并假设 $\Delta=10$ ，我们可以通过上述公式计算得到损失函数值为：
$$L\_i=max(0,−7−13+10)+max(0,11−13+10)$$

我们可以看到第一个max函数求得的值为0，我们可以理解为对第一类的打分13和第二类的打分-7之间的距离为20已经超过我们设置的间隔10，所以不需要惩罚，即这一部分计算得到的损失函数值为0；第一类与第三类的打分距离为2，小于设定的间隔10，所以计算得到损失函数为8。通过上诉例子我们发现损失函数就是用来描述我们对预测的不满意程度，如下图所示，如果预测到的真实类别的分数与错误类别的分数之间的距离都大于我们设置的阈值，则损失函数的值为0。
![](http://7xritj.com1.z0.glb.clouddn.com/16-6-20/62536190.jpg)

这种损失函数就称为**hinge loss**，因为$s\_j=w\_j^Tx\_i$ ， $w\_j$为矩阵**W**的第$j$行展成的列向量，所以公式(2)可以写为：


 $$L\_i=\sum \_ {j\neq y\_i}max(0,w\_j^Tx\_i-w\_{y\_i}^Tx\_i+\Delta) (3)$$


## 正则化

上述损失函数用来约束预测打分和真实打分之间的区别，我们好需要一些参数来约束参数矩阵**W**值的大小，L2正则如下所示，会惩罚过大的参数值：

$$R(W)=\sum \_ k \sum \_ lW \_ {k,l}^2$$

所以对整个数据集总的损失函数如下所示：

$$L= \frac {1}{N} \sum \_ i \sum \_ {j \neq y\_i} [max(0, f(x\_i;W) \_ j -f(x\_i;W) \_ { y\_i} +\Delta)]+\lambda\sum \_ k \sum \_ lW \_ {k,l}^2 $$


## 梯度下降

对公式(2)的 $w\_{y\_i}$ 求导，可以得到：

$$\nabla \_ {W \_ {y \_ i}}L\_i =- (\sum \_ {j \neq y\_i}1(w\_j^Tx\_i-w \_ {y\_i}^Tx\_i + \Delta >0))x\_i$$

其中**1**为指示函数，当括号里的条件成立是函数值为1，否则为0。所以上述**对正确类别所对应分类器权值的求导结果就是把错误类别的打分与正确类别打分间距小于阈值的个数再乘以输入数据$x\_i$**。

对 $j\neq y\_i$ 的其他行，求导结果如下所示，也就是如果这一行所对应的滤波器打分相对于正确的类别分数间隔小于阈值，则对这一行求导所得就是 $x\_i$

$$\nabla \_ {W \_ {j}}L\_i =1(w\_j^Tx\_i-w \_ {y\_i}^Tx\_i + \Delta >0)x\_i$$

其中SVM的hinge loss以及梯度计算如下所示：


```python
def svm_loss_vectorized(W, X, y, reg):

  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  N=X.shape[0]
  D=X.shape[1]

  scores = X.dot(W)
  correct_scores = scores[np.arange(N),y]
  margin = np.maximum(np.zeros(scores.shape),scores+1-correct_scores.reshape(N,-1))
  margin[np.arange(N),y] = 0
  loss = np.sum(margin)
  loss /= N
  loss += 0.5 * reg * np.sum(W * W)

  binary = margin
  binary[margin>0] = 1
  row_sum = np.sum(binary, axis=1)
  binary[np.arange(N), y] = -row_sum[np.arange(N)]
  dW = X.T.dot(binary)
  dW /= N
  dW += reg * W

  return loss, dW


```

## Softmax分类器

 分类器相对于SVM分类器来说，增加了一个计算概率的过程，SVM选择得分最大的一类输出，Softmax把所有的得分转换为每一类的概率，如下公式所示：
$$P(y\_i|x\_i;W)=\frac{e^{f \_ {y\_i}}}{\sum\_je^{f\_j}}$$
其中$f\_j$为分类器对每一类的打分。

Softmax 分类器的损失函数为**cross-entropy loss**，如下所示，其实就是正确类别概率取对数再乘以-1。
$$L\_i = -\log\left(\frac{e^{f\_{y\_i}}}{ \sum\_j e^{f\_j} }\right) \hspace{0.5in} \text{或等于} \hspace{0.5in} L\_i = -f\_{y\_i} + \log\sum\_j e^{f\_j}$$

Softmax 和 SVM分类器的联系区别如下图所示：
![](http://7xritj.com1.z0.glb.clouddn.com/16-6-20/8049276.jpg)

cross-entropy loss求导

对$w\_{y\_i}$:
$$\nabla \_ {W \_ {y \_ i}}L\_i =-x\_i+p \_ {y\_i}x\_i$$

对$w\_j(j \neq y\_i)$:
$$\nabla \_ {W \_ {y \_ i}}L\_i =p \_ {j}x\_i$$
其中 $p\_{j}$ 为Softmax分类器输出为第 $j$ 类的概率。

Softmax的cross-entropy loss以及梯度计算如下所示：


```python
def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  N, D = X.shape


  scores = X.dot(W) #(N,C)


  p = np.exp(scores.T)/np.sum(np.exp(scores.T),axis=0)

  p=p.T

  loss = -np.sum(np.log(p[np.arange(N), y]))

  p[np.arange(N), y] = p[np.arange(N), y]-1

  dW = X.T.dot(p)


  loss /=N
  dW /=N
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W

  return loss, dW
```

## 随机梯度下降

在大数据集的训练中，计算所有数据的损失函数只更新一次参数是很浪费的行为。一个通常的做法是计算一批训练数据的梯度然后更新，能用这个方法的是基于所以训练数据都是相关的假设，每一批数据的梯度是所有数据的一个近似估计。


```python
class LinearClassifier(object):

  def __init__(self):
    self.W = None

  def train(self, X, y, learning_rate=1e-3, reg=1e-5, num_iters=100,
            batch_size=200, verbose=False):

    num_train, dim = X.shape
    num_classes = np.max(y) + 1 # assume y takes values 0...K-1 where K is number of classes
    if self.W is None:
      # lazily initialize W
      self.W = 0.001 * np.random.randn(dim, num_classes)

    # Run stochastic gradient descent to optimize W
    loss_history = []
    for it in xrange(num_iters):
      X_batch = None
      y_batch = None

      index=np.random.choice(num_train,batch_size,replace=False)

      X_batch=X[index]
      y_batch = y[index]

      loss, grad = self.loss(X_batch, y_batch, reg)
      loss_history.append(loss)

      self.W -= learning_rate*grad

      if verbose and it % 100 == 0:
        print 'iteration %d / %d: loss %f' % (it, num_iters, loss)

    return loss_history

  def predict(self, X):
    y_pred = np.zeros(X.shape[1])
    scores = X.dot(self.W)

    y_pred = np.argmax(scores,axis=1)
    return y_pred

  def loss(self, X_batch, y_batch, reg):
    pass


class LinearSVM(LinearClassifier):

  def loss(self, X_batch, y_batch, reg):
    return svm_loss_vectorized(self.W, X_batch, y_batch, reg)

class Softmax(LinearClassifier):
  """ A subclass that uses the Softmax + Cross-entropy loss function """

  def loss(self, X_batch, y_batch, reg):
    return softmax_loss_vectorized(self.W, X_batch, y_batch, reg)

```

训练SVM：


```python
svm = LinearSVM()
svm.train(X_train, y_train, learning_rate=1e-3, reg=1e0, num_iters=400,verbose=True)
y_train_pred = svm.predict(X_train)
acc_train = np.mean(y_train == y_train_pred)
y_val_pred = svm.predict(X_val)
acc_val = np.mean(y_val == y_val_pred)
y_test_pred = svm.predict(X_test)
acc_test = np.mean(y_test == y_test_pred)
print train_accuracy, val_accuracy, acc_test
```

    iteration 0 / 400: loss 8.953550
    iteration 100 / 400: loss 0.208299
    iteration 200 / 400: loss 0.287735
    iteration 300 / 400: loss 0.233046
    0.940448569219 0.951388888889 0.952777777778


## 对线性分类器的直观解释

参数**W**的每一行可以理解为一个图像模板，每个类别的得分就是输入图片与每一行的图片模板**内积**的结果，看输出的图片最符合哪个图片模板，也就是最可能符合哪一类。也就是说通过训练，分类器的每一行学习到了每类图片的模板，如下图所示，线性SVM分类器学习到的每一类数字的模板图片。


```python

w = svm.W[:-1,:] # strip out the bias
#print w
w = w.reshape(8, 8, 10)
w_min, w_max = np.min(w), np.max(w)
for i in xrange(10):
  plt.subplot(2, 5, i + 1)

  #Rescale the weights to be between 0 and 255
  wimg = 255.0 * (w[:, :, i].squeeze() - w_min) / (w_max - w_min)
  #wimg = w[:, :, i]
  plt.imshow(wimg.astype('uint8'))
  plt.axis('off')
  plt.title(classes[i])
```


![](http://7xritj.com1.z0.glb.clouddn.com/16-6-20/94762256.jpg)



```python
sf = Softmax()
sf.train(X_train, y_train, learning_rate=1e-2, reg=0.5, num_iters=500,verbose=True)
y_train_pred = sf.predict(X_train)
acc_train = np.mean(y_train == y_train_pred)
y_val_pred = sf.predict(X_val)
acc_val = np.mean(y_val == y_val_pred)
y_test_pred = sf.predict(X_test)
acc_test = np.mean(y_test == y_test_pred)
print train_accuracy, val_accuracy, acc_test
```

    iteration 0 / 500: loss 2.301230
    iteration 100 / 500: loss 0.362278
    iteration 200 / 500: loss 0.361298
    iteration 300 / 500: loss 0.352944
    iteration 400 / 500: loss 0.365702
    0.940448569219 0.951388888889 0.95



```python
w = sf.W[:-1,:] # strip out the bias
print w.shape
w = w.reshape(8, 8, 10)
w_min, w_max = np.min(w), np.max(w)
for i in xrange(10):
  plt.subplot(2, 5, i + 1)

  #Rescale the weights to be between 0 and 255
  wimg = 255.0 * (w[:, :, i].squeeze() - w_min) / (w_max - w_min)
  #wimg = w[:, :, i]
  plt.imshow(wimg.astype('uint8'))
  plt.axis('off')
  plt.title(classes[i])
```

    (64, 10)



![](http://7xritj.com1.z0.glb.clouddn.com/16-6-20/30387585.jpg)


## 参考

[CS231n: Convolutional Neural Networks for Visual Recognition. ](http://cs231n.github.io/)
