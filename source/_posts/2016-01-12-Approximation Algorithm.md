---
layout:     post
title:      "Approximation Algorithm"
subtitle:   "Bin Packing&Steiner Tree Problem"
date:       2016-01-12 10:00:00
author:     "Ldy"
header-img: "img/tag-bg1.jpg"
tags:
    - Algorithms
---


# 概念

许多具有卖际意义的问题都是 [NP 完全](http://buptldy.github.io/2016/01/11/NP%E5%AE%8C%E5%85%A8%E9%97%AE%E9%A2%98%E7%9A%84%E4%BB%8B%E7%BB%8D%E5%8F%8A%E8%AF%81%E6%98%8E/)问题。我们不知道如何在多项式时间内求得最优解。但是，这些问题通常又十分重要， 我们不能因此而放弃对它们的求解。即使一个问题是 NP 完全的，也有其求解方法。解决 NP 完全问题至少有三种方法：
<!--more-->
- 如果实际输入数据规模较小，则用指数级运行时的算法就能很好地解决问题；
- 对于一些能在多项式时间内解决的特殊情况，可以把它们单独列出来求解；
- 可以寻找一些能够在多项式时间内得到近似最优解 （near-optimal solution)的方法(最坏情况或平均情况)。

在实际应用中，近似最优解一般都能满足要求， 返回近似最优解的算法就称为近似算法(approximation algorithm)。

**近似比：**

如果对规模为n的任意输人，近似算法所产生的近似解的代价C与最优解的代价C*只差一个因子$\rho (n)$:

$$max(\frac{C}{C\*},\frac{C\*}{C} )\leq \rho (n)$$

则称该近似算法有近似比$\rho (n)$。如果一个算法的近似比达到 $\rho (n)$，则称该算法为$\rho (n)$近似算法。近似比和$\rho (n)$近似算法的定义对求最大化和最小化问题都适用，一个近似算法的近似比不会小于1。

# 一维装箱问题(Bin Packing)

问题如下：

Bin Packing is as follows: Given n items with sizes $a_1, \cdots , a_n ∈ (0, 1]$, find a packing in unit-sized bins that minimizes the number of bins used.

Give a 2-approximation algorithm for this problem and analysis the approximation factor.

装箱问题：有n个物品，每个物品的尺寸在0-1之间，每个箱子的容量为1，问最少要用多少的箱子能把所有的物品装下？

装箱问题可用整数规划描述如下，其中$y_i=1$表示箱子$i$被使用，否则表示没有使用，$x_{ij}=1$表示物品j放入箱子i中。

<img src="http://i.imgur.com/VPewCSm.png" style="display:block;margin:auto"/>


其中约束条件(1)表示：一旦箱子i被使用，放入箱子i中的物品尺寸不能超过箱子的容量1。

约束条件(2)表示：每个物品刚好放入一个箱子中。

由[前文](http://buptldy.github.io/2016/01/11/NP%E5%AE%8C%E5%85%A8%E9%97%AE%E9%A2%98%E7%9A%84%E4%BB%8B%E7%BB%8D%E5%8F%8A%E8%AF%81%E6%98%8E/)已知，整数线性规划问题是NP完全的，即不能找到多项式时间算法来求解，所以需要寻找一种近似算法。

> Next Fit算法：按顺序把物品放进当前箱子，如果放不下，则放下一个。

举例：

|物品|$J_1$|$J_2$|$J_3$|$J_4$|$J_5$|$J_6$|
|-|-|-|-|-|-|-|
|尺寸|0.6|0.7|0.4|0.2|0.8|0.3|

根据Next Fit算法，解如下图所示：

<center>
![](http://i.imgur.com/YDnGdKS.png)
</center>


> 证明：Next Fit是Bin Packing问题近似比为2的近似算法

- 对所有的输入物品序列I有:$NF(I) \leq 2OPT(I)$

	如下图所示，任意考虑两个相邻的箱子，这两个箱子里面的物品的容量肯定要大于1，否则根据Next Fit算法会把这些物品放进第一个箱子，所以两个相邻箱子所占用的空间肯定是大于1的，即有$B_1+B_2>1$，对于$B_3+B_4,\dots$都是这样。因此浪费的空间不达到一半，所以有$NF(I) \leq 2OPT(I)$。

<center>
	![](http://i.imgur.com/A8GpRKX.png)
</center>

- 存在一个输入物品序列I：$NF(I)\geq 2OPT(I)-2$

	考虑长度为n(n为4的倍数)的物品序列I，尺寸大小分别为：

	0.5，2/n，0.5，2/n，...，0.5，2/n

	则最佳装箱策略如下图所示，最少需要(n/4+1)个箱子。
<center>
![](http://i.imgur.com/IPg1QX9.png)
</center>
	Next Fit策略如下图所示，需要(n/2)个箱子
<center>
![](http://i.imgur.com/EkthQ3j.png)
</center>
	所以根据上述证明，Next Fit是Bin Packing问题近似比为2的近似算法。

**复杂度分析**：由于NF算法处理每个物品只检查一个箱子,所以其时间复杂度是线性的,但也正因为如此,使得前面箱子剩余空间再无利用的可能。该算法的时间复杂度是 $O(n)$ , 空间复杂度为 $O(1)$ 。

# Steiner Tree Problem

问题如下：

Given an undirected graph G = (V, E) with edge costs and set T ⊆ V of required vertices, the Steiner Tree Problem is to find a minimum cost tree in G containing every vertex in T (vertices in V −T may or may not be used in T).

Give a 2-approximation algorithm if G is complete and the edge costs satisfy the triangle inequality.

所谓的Steine​​r tree problem是指在一无向图G(V,E)中, 给定一组V的子集合S, 我们要在其中找到一个minimum cost tree, 这个tree 必需包含S中所有的点, 另外也可包含一些非S中的点。这些非S的点我们称之为Steine​​r nodes, S中的点我们称之为terminals。

Steine​​r tree problem 是属于NP-complete 的间题, 代表着我们目前找不到一个算法, 能够在polynomial 的时间内解决这个问题。

**问题详述:**
<center>
![](http://i.imgur.com/TF3lU8i.gif)
</center>
所谓的Steine​​r Tree Problem, 是一组无向图G(V,E)中, 给定一组terminals, 如图一的A和D, 然后我们必需在G上找到一个minimum spanning tree, 这个tree 必需满足下面要求

- 它必需span 所有的terminals
- 它可以包含非terminal 的点, 这些点称之为steine​​r node, 如图1的B, E, F
- 它的total cost必需为最小

在上图中我们可以知道, 如果不能包含非terminal 的点, 则找出来的spanning tree, cost为6, 而且有可能根本找不到这样的tree, 在包含了一些steine​​r node 之后, 所找出的cost为5。
**近似算法**

The Kou Markowsky and Berman algorithm

Input: a undirect graph G(V,E) and a subset S of V.

Output: The minimum cost Steine​​r tree T.

Step1:建构distance graph G1(S, E'). 对每一个E'中的edge (u, v),它的cost等于G中u到v的最短路径的cost

Step2: 找出minimum spaning tree T1 of G1

Step3: 建构G2(V'', E''), 将T1的每一个edge (u, v), 用它在Step1中所找的路径代入.

Step4: 将G2中的cycle去掉.

<center>
  ![](http://i.imgur.com/CohPHo2.png)
</center>



如上图所示, 首先我们先建立一个包含所有terminal 的complete distance graph G1, 然后找出它的minimum spanning tree T1, 然后将原路径代回, 得到G2, 最后将G2 的cycle移去, 得到total cost 为14 的Steine​​r Tree T. 因为此一近似算法为approximate algorithm, 所以它得到的steine​​r tree并一定都是optimum, 此例子的minimum Steine​​r tree的cost为13。

复杂度和近似比：因为需要计算最短路径，所以时间复杂度为$O(M*N^2)$,其中 $|V|=N$ , $\|S\|=M$。

近似比为2，证明可以参考：[http://www.csie.ntu.edu.tw/~kmchao/tree10fall/Steiner.pdf](http://www.csie.ntu.edu.tw/~kmchao/tree10fall/Steiner.pdf "http://www.csie.ntu.edu.tw/~kmchao/tree10fall/Steiner.pdf")

# 参考

演算法设计与分析Term Project：[http://par.cse.nsysu.edu.tw/~homework/algo01/9034811/Report/index.htm](http://par.cse.nsysu.edu.tw/~homework/algo01/9034811/Report/index.htm "演算法设计与分析Term Project")
