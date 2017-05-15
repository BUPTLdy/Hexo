---
layout:     post
title:      "Backtracking Strategies"
subtitle:   " \"解决N皇后问题\""
date:       2016-01-03 10:00:00
author:     "Ldy"
header-img: "img/tag-bg1.jpg"
tags:
    - Algorithms
    - Artificial Intelligence
---

# 回溯策略（backtracking）简介

回溯法采用试错的思想，它尝试分步的去解决一个问题。在分步解决问题的过程中，当它通过尝试发现现有的分步答案不能得到有效的正确的解答的时候，它将取消上一步甚至是上几步的计算，再通过其它的可能的分步解答再次尝试寻找问题的答案。回溯法通常用最简单的递归方法来实现，在反复重复上述的步骤后可能出现两种情况：
<!--more-->	- 找到一个可能存在的正确的答案
- 在尝试了所有可能的分步方法后宣告该问题没有答案

在最坏的情况下，回溯法会导致一次复杂度为指数时间的计算。

# 回溯策略应用：四皇后问题（Four queens puzzle）

四皇后问题：在一个国际象棋中的 的棋盘上放置4个皇后， 为了使其中的任何2个皇后都不能相互“攻击”，希望寻求4个皇后的安全放置位置。 该问题的不能相互“攻击”相当于要求任意两个皇后不能在同一行、同一列或同一斜线上。

用回溯策略解决这一问题：

- 首先放第一颗棋子

<center>
![](http://i.imgur.com/775UH8f.png)
</center>

- 在符合规则的条件下摆放第二课棋子
<center>
![](http://i.imgur.com/OYEeASd.png)
</center>

- 在规则下未找到解时，回溯

<center>
![](http://i.imgur.com/wIxeLH5.png)
</center>

- 根据回溯策略，不断的试探，最终找到解为：

<center>
![](http://i.imgur.com/f1QnRh3.png)
</center>

# 回溯搜索中知识的利用

在回溯策略中，可以通过引入一些与问题有关的信息来加快搜索解的速度。对与N皇后问题来说，引入信息的基本思想是：

尽可能选取划去对角线上位置数最少的
<center>
![](http://i.imgur.com/JCaTEEu.png)
</center>
可以想象，如果把一个皇后放在棋盘的某个位置后，它所影响的棋盘位置数少，那么给以后放置皇后剩下的余地就越大，找到解的可能性也越大。

# 回溯算法存在的问题及解决方案

存在的问题：

	- 某一个分支具有无穷个状态，算法可能落入“深渊”，永远不能回溯
	- 某一个分支上具有环路，搜索在环路中一直进行，同样不能回溯

解决方案：

	- 对搜索深度进行限制，当当前状态的深度达到了限制深度时，算法将进行回溯
	- 记录从初始状态到当前状态的路径，如果出现过此路径，表明出现环路，算法回溯

# 参考

维基百科回溯法：[https://zh.wikipedia.org/wiki/%E5%9B%9E%E6%BA%AF%E6%B3%95](https://zh.wikipedia.org/wiki/%E5%9B%9E%E6%BA%AF%E6%B3%95 "维基百科：回溯法")

四皇后问题：[http://jpkc.onlinesjtu.com/CourseShare/DataStructure/FlashInteractivePage/exp7.htm](http://jpkc.onlinesjtu.com/CourseShare/DataStructure/FlashInteractivePage/exp7.htm "四皇后问题")
