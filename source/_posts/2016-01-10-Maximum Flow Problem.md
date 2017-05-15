---
layout:     post
title:      "Maximum Flow Problem"
subtitle:   "Ford-Fulkerson algorithm & Push-relabel algorithm"
date:       2016-01-10 10:00:00
author:     "Ldy"
header-img: "img/tag-bg1.jpg"
tags:
    - Algorithms
    - Python

---


# 网络流的定义

在图论中，网络流（Network flow）是指在一个每条边都有容量（capacity）的有向图上分配每条路劲流量，使一条边的流量不会超过它的容量。通常在运筹学中，有向图称为网络。顶点称为节点（node）而边称为弧（arc）。一道流必须符合一个结点的进出的流量相同的限制，除非这是一个源点（source）──有较多向外的流，或是一个汇点（sink）──有较多向内的流。一个网络可以用来模拟道路系统的交通量、管中的液体、电路中的电流或类似一些东西在一个结点的网络中游动的任何事物。
<!--more-->

假设 G = (V,E) 是一个有限的有向图，它的每条边$ \ (u,v) \in E $ 都有一个非负值实数的容量$c(u, v)$。如果$(u, v) \not \in E$，我们假设 $c(u, v) = 0$。我们区别两个顶点：一个源点 s 和一个汇点 t 。一道网络流是一个对于所有结点 u 和 v 都有以下特性的实数函数 $f:$：



> 容量限制（Capacity Constraints）：	$f(u, v) \le c(u, v)$一条边的流不能超过它的容量。

> 流量守恒（Flow Conservation）：	除非u = s或u = t，否则 $\sum\_{w \in V} f(u, w) = 0$，中间结点的流入等于流出。


即流守恒意味着： $\sum\_{(u,v) \in E} f(u,v) = \sum\_{(v,z) \in E} f(v,z) $，对每个顶点${v \in V\setminus{s,t}}$。

**最大流问题**就是，给定一个流网络G，一个源节点s，一个汇点t，我们希望找到值最大的一个流。

一个网络流如下图所示：

<center>
![](http://i.imgur.com/a8NFZl3.png)
</center>

# 网络最大流算法

- Ford-Fulkerson算法

残存网络的概念：

边的残存容量（residual capacity）是$ c\_f(u, v) = c(u, v) - f(u, v)$。

定义 $G\_f(V, E\_f)$ 表示剩余网络（residual network），它显示当前网络可用的容量的多少。就算在原网络中由 u 到 v 没有边，在剩余网络仍可能有由 u 到 v 的边。因为残存网络允许相反方向的流抵消，减少由 v 到 u 的流相当于增加由 u 到 v 的流，因为我们是为了求最大流，之前走过的路可能是走错的。

增广路（augmenting path）是一条路径 $(u\_1, u\_2, \dots, u\_k)$，而$u\_1 = s , u\_k = t $ 及$c\_f(u\_i , u\_{i+1})>0$，如果存在增广路，这表示沿这条路径还能够传送更多流。当且仅当剩余网络$G\_f$ 没有增广路时处于最大流。

建立残存网络$\ G\_f$的步骤：

1. $\ G\_f = \ V $ 的顶点
2. 定义如下的 $\ G\_f = \ E\_f$ 的边,对每条边 $\ (x,y) \in E$

	- 若$\ f(x,y) < c(x,y)$，建立容量为$\ c\_f = c(x,y) - f(x,y)$ 的前向边$\ (x,y) \in E\_f$。
	- 若$\ f(x,y) > 0$，建立容量为$\ c\_f =  f(x,y)$ 的后向边$\ (y, x) \in E\_f$。

上图中的残存网络如下图所示：
<center>
![](http://i.imgur.com/RS7lQTw.png)
</center>
最小割的概念：

割的定义： 一个s-t 的割 C = (S, T) 把 所有的结点集合V分成两部分S和T，其中源节点s ∈ S 汇点 t ∈ T. 割 C 的集合表示如下：

$$\{(u,v)\in E\:\ u\in S,v\in T\}$$

**当割中的边被移掉时,则从源节点到汇结点的流量为0**.

割容量的定义：

$$c(S,T)=\sum \nolimits\_{(u,v)\in S\times T}c\_{uv}$$


>最大流最小割定理：当残存网络中不含有任何增广路径时，网络中的流量f最大且等于最小割容量。

如下图所示，网络的最大流为7，最小割由图中虚线组成，其中最小割的容量也为7.
<center>
![](http://i.imgur.com/7Q718mT.png)
</center>
Ford-Fulkerson算法求网络的最大流就是在每次的迭代中，寻找某条增广路径p，然后使用p来对流f进行修改，直到残存网络中不含有任何增广路径时，求得最大流f。

Ford-Fulkerson算法伪代码：
<center>
![](http://i.imgur.com/zsQkoBR.png)
</center>
[Ford-Fulkerson算法Python实现](https://github.com/BUPTLdy/Algorithms/tree/master/Ford-Fulkerson)

- Push-relabel algorithm

Push-Relabel系的算法普遍要比Ford-Fulkerson系的算法快，但是缺点是相对难以理解。详细内容可以参考[维基百科 Push–relabel maximum flow algorithm](https://en.wikipedia.org/wiki/Push%E2%80%93relabel_maximum_flow_algorithm#Concepts)

[Push-relabel 算法Python实现](https://github.com/BUPTLdy/Algorithms/tree/master/Push-relabel)

# 网络最大流问题建模

问题如下：

Support the you are a matchmaker and there are some boys and girls. Since the boys are alway more than girls, you can assume that if a girl express her love to a boy , the boy will always accept her. Now you know every girl’s thought(a girl may like more than one boy) and you want to make as much pairs as you can. show that you can do this using maximum flow algorithm.

题目意思大概是有一些男孩和女孩。男孩的数量大于等于女孩的数量，如果女孩向男孩表达爱意，男孩必定接受，假设你是个媒婆，而且你知道女孩们喜欢那个男孩(一个女孩可能同时喜欢多个男孩)，要你用最大流的方法求最多能匹配多少对？

问题分析：最终一个女孩肯定只能和一个男孩配对，为了简单分析，我们假设有3个女孩$\{G_1,G_2,G_3\}$，3个男孩$\{B_1,B_2,B_3\}$,并且已知$G_1$喜欢$B_1$和$B_2$，$G_2$喜欢$B_2$,$G_3$喜欢$B_3$，我们可以构造出如下图所示的网络流：
<center>
![](http://i.imgur.com/9hAGUgE.png)
</center>
求出上图所示网络的的最大流就是最大的匹配对数，根据这个思路，这个题抽象成最大流问题为：

	- 源节点，汇点，以及每个女孩男孩都构成一个节点
	- 如果某个女孩喜欢某些男孩，则把这个女孩和那些男孩相连
	- 把源节点和每个女孩相连，汇结点和每个男孩相连
	- 网络中所有相连边的容量为1

# 参考

维基百科 网络流：[https://zh.wikipedia.org/wiki/%E7%BD%91%E7%BB%9C%E6%B5%81](https://zh.wikipedia.org/wiki/%E7%BD%91%E7%BB%9C%E6%B5%81 "维基百科 网络流")

Wikipedia Max-flow min-cut theorem:[https://en.wikipedia.org/wiki/Max-flow_min-cut_theorem](https://en.wikipedia.org/wiki/Max-flow_min-cut_theorem "Wikipedia Max-flow min-cut theorem")
