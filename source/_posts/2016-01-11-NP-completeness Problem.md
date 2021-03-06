---
layout:     post
title:      "NP-completeness Problem"
subtitle:   "介绍及证明"
date:       2016-01-11 10:00:00
author:     "Ldy"
header-img: "img/tag-bg1.jpg"
tags:
    - Algorithms
---


## NP完全性

到目前为止，我们讨论的几乎都是**多项式时间算法**：对于规模n的输入，在最坏情况下的运行时间是$O(n^k)$,其中k为某一确定常数。但还有很多问题在多项式时间内并不能求解，根据能否在多项式时间求解，定义如下几类问题：
<!--more-->
- P: 能在多项式时间内解决的问题
- NP: 不能在多项式时间内解决或不确定能不能在多项式时间内解决，但能在多项式时间验证的问题
- NPC: NP完全问题，所有NP问题在多项式时间内都能约化(Reducibility)到它的NP问题，即解决了此NPC问题，所有NP问题也都得到解决。
- NP hard:NP难问题，所有NP问题在多项式时间内都能约化(Reducibility)到它的问题(不一定是NP问题)。

如果任何NP完全问题是可以多项式求解的，则P=NP，目前还不能证明P是否等于NP，这几个问题的关系如下：
<center>
![](http://i.imgur.com/mAyE7SM.png)
</center>
规约的概念：若我们拥有一个已证明难以解决的问题，我们又获得另一个相似的新问题。我们可合理推想此新问题亦是难以解决的。我们可由下列谬证法得证：若此新问题本质上容易解答，且若我们可展示每个旧问题的实例可经由一系列转换步骤变成新问题的实例，则旧问题便容易解决，因此得到悖论。因此新问题可知亦难以解决。

如何证明某个问题是NP完全的？

如果我们有一个已经证明的NP完全问题，如果我们可以把已证明的NP完全问题的任何实例都能多项式的规约到要要证明问题的实例，则能证明这个问题是NP完全的。即如果我们要证明一个问题A是NPC问题，**则只需要首先证明他是NP问题，然后只要找一个你所知道的NPC问题规约到A即可。**

## 常见的NPC问题
### 布尔可满足性问题（SAT）

对于一个确定的逻辑电路，是否存在一种输入使得输出为真。是第一个被证明的NPC问题，直观的看出这应该是一个NPC问题，因为当电路有k个输入，就会有$2^k$种情况的不同取值。
### 3SAT

3和取范式：公式中每个字句都恰好有三个不同的‘文字’，例如，布尔公式：

$$(l\_1 ∨ l\_2 ∨ x\_2) ∧ (¬x\_2 ∨ l\_3 ∨ x\_3) ∧ (¬x\_3 ∨ l\_4 ∨ x\_4) ∧ ... ∧ (¬x\_{n − 3} ∨ l\_{n − 2} ∨ x\_{n − 2}) ∧ (¬x\_{n − 2} ∨ l\_{n − 1} ∨ l\_n)$$
3SAT问题就是满足3和取范式的布尔公式是否可满足，3SAT问题可由SAT问题规约而来。

### 分团问题（clique problem）

无向图中的团是图中所有顶点的一个子集，团中的每一对顶点之间都有一条边相连，即一个团就是无向图中的一个完全子图。分团问题就是要寻找图中规模最大的团，判定条件：在图中是否存在一个给定规模为k的团。
### 独立集问题（Independent Set）

独立集：如果有一个顶点集合S，S中的任意两个顶点之间都没有边相连，则称S为一个独立集。
独立集问题和分团问题可相互规约，因为存在一个大小是k以上的分团，等价于它的补图中存在一个大小是k以上的独立集。
补图：一个图G的补图（complement）或者反面（inverse）是一个图有着跟G相同的点，而且这些点之间有边相连当且仅当在G里面他们没有边相连。在制作图的时候，你可以先建立一个有G所有点的完全图，然后清除G里面已经有的边来得到补图，这里的补图并不是图本身的补集。
### 顶点覆盖问题（Vertex Cover）

图的顶点覆盖是一些顶点的集合，使得图中的每一条边都至少接触集合中的一个顶点，如下图所示，图中红色顶点可以覆盖图中所有的边。寻找最小的顶点覆盖的问题称为顶点覆盖问题，它是一个NP完全问题。

<center>
![](http://i.imgur.com/9wD7p7i.png)
</center>

### 集合覆盖问题

给定全集$\mathcal{U}$，以及一个包含n个集合且这n个集合的并集为全集的集合$\mathcal{S}$。集合覆盖问题要找到$\mathcal{S}$的一个最小的子集，使得他们的并集等于全集。
例如$\mathcal{U} = \{1, 2, 3, 4, 5\}，\mathcal{S} = \{\{1, 2, 3\}, \{2, 4\}, \{3, 4\}, \{4, 5\}\}$，虽然$\mathcal{S}$中所有元素的并集是$\mathcal{U}$，但是我们可以找到$\mathcal{S}$的一个子集$\{\{1, 2, 3\}, \{4, 5\}\}$，我们称其为一个集合覆盖。
集合覆盖问题的决定性问题为，给定$(\mathcal{U},\mathcal{S})$和一个整数k，求是否存在一个大小不超过k的覆盖。集合覆盖的最佳化问题为给定$(\mathcal{U},\mathcal{S})$，求使用最少的集合的一个覆盖。

### 子集合问题（subset-sum problem）

给定一个正整数的有限集S和一个整数目标t>0,求是否存在S的一个子集，使得其元素之和为t。

### 3 Coloring

3Col is the problem of deciding whether there is a legal 3-Coloring of a graph (all edges bichromatic).

### 哈密顿回路（Hamiltonian Cycle）

G=(V,E)是一个图，若G中一条通路通过每一个顶点一次且仅一次，称这条通路为哈密尔顿通路。若G中一个圈通过每一个顶点一次且仅一次，称这个圈为哈密尔顿圈。若一个图存在哈密尔顿圈，就称为哈密尔顿图。

## NPC问题的证明

问题如下：

Given an integer m × n matrix A and an integer m-vector b, the Integer programming problem asks whether there is an integer n-vector x such that Ax ≥ b. Prove that Integer-programming is in NP-complete.

证明整数线性规划是NP完全的。

假设上面所提到的常见的NPC问题使我们已知的NPC问题

我们可以把上面提到的顶点覆盖问题（Vertex Cover）规约成整数规划问题，举个简单的例子，简单无向图如下所示：
<center>
![](http://i.imgur.com/GWPD8RB.png)
</center>
很显然就能看出顶点2能够覆盖所有的边，现在我们来讨论这个问题怎么用线性规划来表示，用$y\_i$取1或0来表示是否选择结点i来覆盖边，则该问题用整数线性规划建模如下：
{% raw %}
$$\begin{align*}
         \min y_1+y_2+y_3 \\
         y_1 + y_2 & \ge 1 && \\
         y_2 + y_3 & \ge 1 && \\
         y_1,y_2,y_3 & \ge 0 && \\
         y_1,y_2,y_3 & \in \mathbb{Z} &&
\end{align*}$$
{% endraw %}
如果网络中有结点之间相连，则构成一个不等式约束。从上面例子来看顶点覆盖问题能够规约成整数线性规划问题，一般来说，网络中有多少个结点对应整数线性规划有多少个变量，如果两个结点之间有边相连，则对应一个约束条件，最小化顶点覆盖则对应最小化所有变量之和。所以对无向图G=(V,E),定义整数线性规划如下：
{% raw %}
$$\begin{align*}
         \min \sum_{v \in V} y_v \\
         y_v + y_u & \ge 1 && \forall uv \in E\\
         y_v & \ge 0 && \forall v \in V\\
         y_v & \in \mathbb{Z} && \forall v \in V
\end{align*}$$
{% endraw %}
