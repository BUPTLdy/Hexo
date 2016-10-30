---
layout:     post
title:      "Linear Programming"
subtitle:   "单纯形法 &  线性规划建模"
date:       2016-01-09 10:00:00
author:     "Ldy"
header-img: "img/tag-bg1.jpg"
tags:
    - Algorithms
    - Python
---

# 什么是线性规划

线性规划（Linear Programming，简称LP）是指目标函数和约束条件皆为线性函数的最优化问题。

线性规划问题的常用的最直观形式是标准型。标准型包括以下三个部分：
<!--more-->
- 一个需要极大化的线性函数，例如：

	$$c\_1 x\_1 + c\_2 x\_2$$

- 以下形式的问题约束，例如：

	$$a\_{11} x\_1 + a\_{12} x\_2 \le b\_1$$

	$$a\_{21} x\_1 + a\_{22} x\_2 \le b\_2$$

	$$a\_{31} x\_1 + a\_{32} x\_2 \le b\_3$$

- 非负变量，例如：

	$$x\_1 \ge 0 $$

	$$x\_2 \ge 0 $$

几个概念和定理介绍：

可行域：下图中蓝色部分的点都是线性规划问题的解(可行解)，蓝色区域是可行解的集合，称为可行域。

<center>
![](http://i.imgur.com/0QzN1q2.png)
</center>

基可行解：出现在可行域顶点的可行解

> 定理1 若线性规划问题存在可行域，则其可行域是[凸集](https://zh.wikipedia.org/wiki/%E5%87%B8%E9%9B%86)
>
> 定理2 若可行域有界，线性规划问题的目标函数一定可以在其可行域的顶点上达到最优

通过上述定理得知，线性规划问题的解一定出现在可行域的顶点上，所以可以通过枚举所有的基可行解来找到最优解，但当变量个数很多时，这种办法是行不通的。

# 单纯形法

单纯形法是求解线性规划问题的通用方法。单纯形是美国数学家G.B.丹齐克于1947年首先提出来的。它的理论根据是：线性规划问题的可行域是$n$维向量空间$R\_n$中的多面凸集，其最优值如果存在必在该凸集的某顶点处达到。顶点所对应的可行解称为基本可行解。

单纯形法的基本思想是：先找出一个基本可行解，对它进行鉴别，看是否是最优解；若不是，则按照一定法则转换到另一改进的基本可行解，再鉴别；若仍不是，则再转换，按此重复进行。因基本可行解的个数有限，故经有限次转换必能得出问题的最优解。如果问题无最优解也可用此法判别。

单纯形法的一般解题步骤可归纳如下：

①把线性规划问题的约束方程组表达成典范型方程组，找出基本可行解作为初始基可行解。

②若基本可行解不存在，即约束条件有矛盾，则问题无解。

③若基本可行解存在，从初始基本可行解作为起点，根据最优性条件和可行性条件，引入非基变量取代某一基变量，找出目标函数值更优的另一基本可行解。

④按步骤③进行迭代,直到对应检验数满足最优性条件（这时目标函数值不能再改善），即得到问题的最优解。

⑤若迭代过程中发现问题的目标函数值无界，则终止迭代。

过程如下图所示：

<center>
![](http://i.imgur.com/UfuthIQ.png)
</center>

[单纯形法求解-动态演示](http://wenku.baidu.com/view/0edfb06aaf1ffc4ffe47acec.html)

单纯形法伪代码：

<center>
![](http://i.imgur.com/lhxj168.png)
</center>
[单纯形法Python实现](https://github.com/BUPTLdy/Algorithms/tree/master/Simplex%20Algorithm)

解线性规划问题也一些很好的工具，比如[GLPK](https://www.gnu.org/software/glpk/)和[Gurobi](http://www.gurobi.com/)等。

# 线性规划解决实际问题

问题如下：

with human lives at stake, an air traffic controller has to schedule the airplanes that are landing at an airport in order to avoid airplane collision. Each airplane $i$ has a time window $[s\_i,t\_i]$ during which it can safely land. You must compute the exact time of landing for each airplane that respects these time windows. Furthermore, the airplane landings should be stretched out as much as possible so that the minimum time gap between successive landings is as large as possible. For example, if the time window of landing three airplanes are [10:00-11:00], [11:20-11:40], [12:00-12:20], and they land at 10:00, 11:20, 12:20 respectively, then the smallest gap is 60 minutes, which occurs between the last two airplanes. Given n time windows, denoted as [s_1,t_1], [s_2,t_2], · · ·, [s_n,t_n] satisfying s_1 <t_1 < s_2 < t_2 < · · · < s_n < t_n, you are required to give the exact landing time of each airplane, in which the smallest gap between successive landings is maximized.

	Please formulate this problem as an LP.

题目的大概意思是每架飞机都只能在自己固定的时间窗内降落，为了安全起见两架飞机之间的降落时间间隔越大越好，然后给你n架飞机的降落时间窗口，要求n架飞机的最小降落间隔的最大值。

这个问题的建模起来很简单，令$x\_i$表示第$i$架飞机的降落时间，则需要满足约束条件：

$$s\_i\leq x\_i \leq t\_i$$

然后我们的目标是要求最小间隔的最大值，所以我们的目标函数为：

$$max(min(x\_{i+1}-x\_i))$$

那么现在问题来了，我们上面所说的线性规划的标准形式是不包括既有max又有min的，所以我们需要把这个min去掉，我们可以通过引入一个新变量，如果有$y\leq x\_{i+1}-x\_{i}$，那么$y$不就是$x\_{i+1}-x\_{i}$的最小值吗？

所以最终我们可以把问题形式化为:

{% raw %}
$$  \begin{align*}
    &max~y \\
    s.t.~ &s_i\leq x_i \leq t_i i=1,2,3 \cdots n \\
    &y \leq x_{i+1}-x_i & i=1,2,3 \cdots n
  \end{align*}$$
{% endraw %}
# 参考

维基百科 线性规划：[https://zh.wikipedia.org/wiki/%E7%BA%BF%E6%80%A7%E8%A7%84%E5%88%92](https://zh.wikipedia.org/wiki/%E7%BA%BF%E6%80%A7%E8%A7%84%E5%88%92 "维基百科 线性规划")

百度百科 单纯形法：[http://baike.baidu.com/subview/471090/471090.htm](http://baike.baidu.com/subview/471090/471090.htm "百度百科 单纯形法")
