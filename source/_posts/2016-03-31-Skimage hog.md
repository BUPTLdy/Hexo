---
layout:     post
title:      "Compute the HOG descriptor by skimage "
subtitle:   "skimage hog函数解析"
date:       2016-03-31 11:00:00
author:     "Ldy"
header-img: "img/tag-bg1.jpg"
tags:
    - skimage
    - HOG
---

## HOG简介

方向梯度直方图（英语：Histogram of oriented gradient，简称HOG）是应用在计算机视觉和图像处理领域，用于目标检测的特征描述器。这项技术是用来计算局部图像梯度的方向信息的统计值。这种方法跟边缘方向直方图（edge orientation histograms）、尺度不变特征变换（scale-invariant feature transform descriptors）以及形状上下文方法（ shape contexts）有很多相似之处，但与它们的不同点是：HOG描述器是在一个网格密集的大小统一的细胞单元（dense grid of uniformly spaced cells）上计算，而且为了提高性能，还采用了重叠的局部对比度归一化（overlapping local contrast normalization）技术。

<center>
![](http://7xritj.com1.z0.glb.clouddn.com/16-3-31/40452258.jpg)
</center>


<!--more-->


## 函数形式

```python
skimage.feature.hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3, 3), visualise=False, transform_sqrt=False, feature_vector=True, normalise=None)
```
## hog函数实现的主要步骤

- 图像归一化（可选）
- 计算x和y方向的梯度，包括大小和方向
- 计算梯度柱状图
- 对块状区域进行归一化处理
- 得到一个一维的特征向量

具体有关hog特征计算流程可参考：[《Histograms of Oriented Gradients for Human Detection》论文笔记](http://buptldy.github.io/2016/03/31/2016-03-31-HOG%E8%AE%BA%E6%96%87%E6%80%BB%E7%BB%93/)

## hog函数参数解释

### 传入参数
image : (M, N) ndarray

传入要进行hog特征计算的灰度图

orientations : int

设置方向梯度直方图的箱子个数


pixels_per_cell : 2 tuple (int, int)

设置每个单元的像素

cells_per_block : 2 tuple (int,int)

设置每个区块的单元数

visualise : bool, optional

设置是否返回可视化的hog特征

transform_sqrt : bool, optional

Apply power law compression to normalise the image before processing. DO NOT use this if the image contains negative values. Also see notes section below.
feature_vector : bool, optional

Return the data as a feature vector by calling .ravel() on the result just before returning.
normalise : bool, deprecated

The parameter is deprecated. Use transform_sqrt for power law compression. normalise has been deprecated.


### 返回参数
newarr : ndarray

返回得到的一维hog特征


hog_image : ndarray (if visualise=True)

hog特征的可视化图像

## hog函数举例


```python
import matplotlib.pyplot as plt

from skimage.feature import hog
from skimage import data, color, exposure


image = color.rgb2gray(data.astronaut())

fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualise=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

ax1.axis('off')
ax1.imshow(image, cmap=plt.cm.gray)
ax1.set_title('Input image')
ax1.set_adjustable('box-forced')

# Rescale histogram for better display
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.02))

ax2.axis('off')
ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
ax2.set_title('Histogram of Oriented Gradients')
ax1.set_adjustable('box-forced')
plt.show()

```

<center>
![](http://7xritj.com1.z0.glb.clouddn.com/16-3-31/40452258.jpg)
</center>
