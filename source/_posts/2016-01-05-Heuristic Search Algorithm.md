---
layout:     post
title:      "Heuristic Search Algorithm"
subtitle:   "A*算法 八数码问题"
date:       2016-01-05 10:00:00
author:     "Ldy"
header-img: "img/tag-bg1.jpg"
tags:
    - Algorithms
    - Artificial Intelligence
---

# 什么是启发式搜索
[无信息图搜索](http://buptldy.github.io/2016/01/04/%E5%9B%BE%E6%90%9C%E7%B4%A2/)一般需要产生大量的节点，因而效率较低。为提高效率，可以使用一些问题相关的信息，以减小搜索量，这些信息就称为启发式信息。使用启发式信息指导的搜索过程称为启发式搜索，所以启发式图搜索与无信息图搜索之间的区别就是启发式图搜索在OPEN表的排序过程中使用了与问题有关的知识。
<!--more-->
在启发式搜索过程中对OPEN表进行排序，就需要定义一个评价函数f(n)，对当前的搜索状态进行评估，找出一个*最有希望*的节点来扩展。
# A算法
A算法是一种典型的启发式搜索算法，其基本思想为：定义一个评价函数f(n)，对当前的搜索状态进行评估，找出一个*最有希望*的节点来扩展。
评价函数的形式为：

							f(n)=g(n)+h(n)

其中n是被评价的结点。

为了了解f(n),g(n),h(n)的含义，我们先来介绍一下几个函数的定义：

- g\*(n):从s（初始结点）到n的最短路径的耗散值（相当于一条路径的费用，代价）
- h\*(n):从n到g(目标结点)的最短路径的耗散值
- f\*(n)=g\*(n)+h*(n)：从s经过n到g的最短路径的耗散值

g(n)、 h(n)、 f(n)分别是g\*(n)、 h\*(n)、 f\*(n)的估计值,是一种预测。A算法就是利用这种预测，来达到搜索的目的。**它每次按照f(n)值的大小对OPEN表中的元素进行排序，f值小的放前面，f值大的放后面**，这样每次在扩展结点时，总是选择当前f值最小的结点来优先扩展。

要想根据f对OPEN表中的结点排序，就需要计算f(n),g(n)和h(n)的值，根据搜索结果，g(n)就是初始结点s到结点n这条路径的耗散值；而h(n)依赖于启发信息，取决于具体的问题，通常称其为启发函数。

## A算法举例：八数码问题（Eight-Puzzle）

八数码问题也称为九宫问题。在3×3的棋盘，摆有八个棋子，每个棋子上标有1至8的某一数字，不同棋子上标的数字不相同。棋盘上还有一个空格，与空格相邻的棋子可以移到空格中。要求解决的问题是：给出一个初始状态和一个目标状态，找出一种从初始转变成目标状态的移动棋子步数最少的移动步骤。

<center>
![](http://i.imgur.com/XGw5X8W.png)
</center>

设评价函数f(n)形式如下：

							f(n)=d(n)+W(n)

其中，d(n)代表结点的深度，在单位耗散的情况下g(n)=d(n);取h(n)=W(n)表示以‘不在位’棋子个数作为启发函数的度量。如上图所示，初始状态和目标状态相比，初始状态中的数字“1”，“2”，“6”，“8”不在目标状态的位置上，所以初始状态的h值为4。

使用这种评价函数的搜索树如下所示，图中括弧中的数字表示该结点的评价函数值f;带圆圈的数字表示扩展结点的顺序。

<center>
![](http://i.imgur.com/8DgjWdD.png)
</center>

根据目标结点L返回到s的指针，可得解路径为S(4)，B(4)，E(5)，I(5)，K(5)，L(5)。
# A*算法

最佳图搜索算法A\*(optimal search)，在A算法中，如果有h(n)<=h\*(n),则把这个算法称为A\*算法。当问题有解时，**A\*算法一定能找到一条到达目标结点的最佳路径**。例如，当h(n)恒为零时，满足条件，此时若取g为深度值，则算法等同于宽度优先算法，在[无信息图搜索](http://buptldy.github.io/2016/01/04/%E5%9B%BE%E6%90%9C%E7%B4%A2/)中已提到过，宽度优先算法能够找到一条到目标结点的最短路径。

在使用A\*算法求解问题时，定义的启发函数h，在满足A\*的条件下，应尽可能的大一点，使其接近h\*，这样才能提高搜索的效率，当h=h\*时，搜索的效率最高。

对于八数码问题，取h(n)=W(n),容易看出，尽管我们不知道h\*(n)具体为多少，但是它肯定至少要移动W(n)步才能达到目标状态，因为W(n)为此时和目标状态不相同的数字个数，所以有h(n)<=h\*(n),满足A\*算法条件，所以上述A算法的例子也是A\*算法。


# A*算法的改进

在A\*算法中，扩展一个节点时，对已经在OPEN表或CLOSED表中的子节点，要调整指针，花时间和精力。如果在扩展节点n时，就已经找到了从根节点开始到它的最优路径，则不必调整指针, 可以大大提高效率。如果满足单调性限制，则可实现此愿望。

如果对每一个节点$n_i$以及它的后继节点$n_j$，满足：

$$h(n_i) - h(n_j) ≤ k(n_i,n_j)$$

则称启发式函数满足单调性限制。

如果A\*满足单调性限制，则当它选择节点n扩展时，就已经发现了通向节点n的最佳路径,则不必进行结点的指针修正操作，因而改善了A\*的效率。