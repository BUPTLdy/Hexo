---
layout:     post
title:      "Basic Sorting Algorithms Implemented In Python"
subtitle:   "Python实现常见排序算法"
date:       2016-05-09 12:00:00
author:     "Ldy"
header-img: "img/tag-bg1.jpg"
tags:
    - Algorithms
    - Python
---

<center>
![](http://7xritj.com1.z0.glb.clouddn.com/16-5-9/51291175.jpg)
</center>
<!--more-->

# 冒泡排序

冒泡排序比较简单，主要过程如下：
- 比较相邻的元素。如果第一个比第二个大，就交换他们两个。
- 对每一对相邻元素作同样的工作，从开始第一对到结尾的最后一对。这步做完后，最后的元素会是最大的数。
- 针对所有的元素重复以上的步骤，除了最后一个。
- 持续每次对越来越少的元素重复上面的步骤，直到没有任何一对数字需要比较。

```python
def BubbleSort(array):
    for i in xrange(len(array)):
        for j in xrange(len(array)-1):
            if array[j]>array[j+1]:
                array[j],array[j+1]=array[j+1],array[j]
    return array
```

# 选择排序

选择排序（Selection sort）是一种简单直观的排序算法。它的工作原理如下。首先在未排序序列中找到最小（大）元素，存放到排序序列的起始位置，然后，再从剩余未排序元素中继续寻找最小（大）元素，然后放到已排序序列的末尾。以此类推，直到所有元素均排序完毕。

```python
def SelectionSort(array):
    for i in xrange(len(array)):
        min_index=i
        for j in xrange(i+1,len(array)):
            if array[j]<array[min_index]:
                min_index=j
        array[i],array[min_index]=array[min_index],array[i]
    return array
```

# 插入排序

插入排序（英语：Insertion Sort）是一种简单直观的排序算法。它的工作原理是通过构建有序序列，对于未排序数据，在已排序序列中从后向前扫描，找到相应位置并插入。插入排序在实现上，通常采用in-place排序（即只需用到O(1)的额外空间的排序），因而在从后向前扫描过程中，需要反复把已排序元素逐步向后挪位，为最新元素提供插入空间。

```python
def InsertionSort(array):
    for i in xrange(1,len(array)):
        temp=array[i]
        for j in xrange(i,-1,-1):
            if temp>array[j-1]:
                break
            else:
                array[j]=array[j-1]
        array[j]=temp
    return array
```

# 归并排序

归并排序（英语：Merge sort，或mergesort），是创建在归并操作上的一种有效的排序算法，效率为O(n log n)。1945年由约翰·冯·诺伊曼首次提出。该算法是采用分治法（Divide and Conquer）的一个非常典型的应用，且各层分治递归可以同时进行。

有关归并排序中的详细内容可以参考[分治策略中的归并排序](http://buptldy.github.io/2016/01/06/2016-01-06-%E5%88%86%E6%B2%BB%E7%AD%96%E7%95%A5[Divide%20and%20Conquer]/)

```python
def MergeSort(array):
    n=len(array)
    if n<=1:
        return array
    else:
        n=n/2
        left=MergeSort(array[0:n])
        right=MergeSort(array[n:])
        return Merge(left,right)

def Merge(left,right):
    array=[]
    while len(left)>0 and len(right)>0:
        if left[0]<right[0]:
            array.append(left[0])
            del left[0]
        else:
            array.append(right[0])
            del right[0]
    if len(left)>0:
        array.extend(left)
    if len(right)>0:
        array.extend(right)
    return array
```


# 快速排序

快速排序使用分治法（Divide and conquer）策略来把一个序列（list）分为两个子序列（sub-lists）。

步骤为：
- 从数列中挑出一个元素，称为"基准"（pivot），重新排序数列，所有元素比基准值小的摆放在基准前面，所有元素比基准值大的摆在基准的后面（相同的数可以到任一边）。在这个分区结束之后，该基准就处于数列的中间位置。这个称为分区（partition）操作。
- 递归地（recursive）把小于基准值元素的子数列和大于基准值元素的子数列排序。
- 递归的最底部情形，是数列的大小是零或一，也就是永远都已经被排序好了。虽然一直递归下去，但是这个算法总会结束，因为在每次的迭代（iteration）中，它至少会把一个元素摆到它最后的位置去。

```python
def QuickSort(array):
    if len(array)<=1:
        return array
    pivot=array[0]
    left=[x for x in array[1:]if x<pivot ]
    right=[x for x in array[1:] if x>=pivot]
    return QuickSort(left)+[pivot]+QuickSort(right)

```

# 堆排序

在堆的数据结构中，堆中的最大值总是位于根节点。堆中定义以下几种操作：

- 最大堆调整（Max_Heapify）：将堆的末端子节点作调整，使得子节点永远小于父节点
- 创建最大堆（Build_Max_Heap）：将堆所有数据重新排序
- 堆排序（HeapSort）：移除位在第一个数据的根节点，并做最大堆调整的递归运算

堆排序可以参考这篇博文：[http://www.cnblogs.com/cj723/archive/2011/04/22/2024269.html]（http://www.cnblogs.com/cj723/archive/2011/04/22/2024269.html）

```python

def heap_sort(array):

def sift_down(start, end):
"""最大堆调整"""
root = start
while True:
    child = 2 * root + 1    #左子节点
    if child > end:         #如果没有子节点退出
        break
    if child + 1 <= end and array[child] < array[child + 1]: #如果左子节点值小于右子节点
        child += 1                             #下标由左子节点更换为右子节点
    if array[root] < array[child]:             #如果父节点小与子节点，则值相互交换
        array[root], array[child] = array[child], array[root]
        root = child                           #对发生变化的子节点向下递归，重复上述过程
    else:
        break

# 创建最大堆
for start in xrange((len(array) - 2) // 2, -1, -1):#从最后一个非叶子节点开始构造最大堆
sift_down(start, len(array) - 1)

# 堆排序
for end in xrange(len(array) - 1, 0, -1):
array[0], array[end] = array[end], array[0] #把最大值放在最后
sift_down(0, end - 1)                      #除最大值之外的继续构造最大堆
return array
```
