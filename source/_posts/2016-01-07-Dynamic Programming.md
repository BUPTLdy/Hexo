---
layout:     post
title:      "Dynamic Programming"
subtitle:   "Money robbing &  Maximum profit"
date:       2016-01-07 10:00:00
author:     "Ldy"
header-img: "img/tag-bg1.jpg"
tags:
    - Algorithms
    - CPP
---

# 动态规划(dynamic programming)

动态规划与[分治方法](http://buptldy.github.io/2016/01/06/%E5%88%86%E6%B2%BB%E7%AD%96%E7%95%A5%5BDivide%20and%20Conquer%5D/)相似，都是通过组合子问题的接来求解原问题，分治方法是将问题划分为互不相交的子问题，递归的求解子问题，再将它们的解组合起来，求出原问题的解。动态规划与之相反，应用于子问题重叠的情况，即不同的子问题具有公共的子子问题。在这种情况下，分治算法会反复的求解这些公共子问题，而动态规划对每个子问题只求解一次并保存结果，从而无需重复求解。
<!--more-->
通常动态规划被用来求解[最优化问题](https://en.wikipedia.org/wiki/Optimization_problem)，即在可行解中寻找最优解。

设计一个动态规划算法通常有如下4个步骤：

	- (1).刻画一个最优解的结构特征
	- (2).递归地定义最优解的值
	- (3).采用自底向上的方法计算最优解的值
	- (4).利用计算出的信息构造最优解

如果我们只需要得到这个最优解的结果，而不关注这个解是怎么得来的，则可以忽略步骤(4)。

上面叙述了动态规划方法的步骤，但是什么问题才能够使用动态规划法求解？使用动态规划方法求解的最优化问题应该具备两个要素：最优子结构和子问题重叠。

- 最优子结构

如果一个问题的最优解包含其子问题的最优解，则称此问题具有最优子结构性质。在动态规划方法中，我们通常自底向上地使用最优子结构，即首先求得子问题的最优解，然后求原问题的最优解。**原问题的最优解的代价通常就是子问题最优解的代价再加上由此次选择直接产生的代价。**

- 子问题重叠

如果递归算法反复求解相同的子问题，就称最优化问题具有重叠子问题。动态规划算法通常这样利用重叠子问题性质：**对每个子问题求解一次，将解存入一个表中，当再次需要这个子问题时直接查表，每次查表的代价为常量时间。**

# 问题举例：Money robbing

问题如下所示：

A robber planning to rob houses along a street. Each house has a certain amount of money stashed, the only constraint stopping you from robbing each of them is that adjacent houses have security system connected and it will automatically contact the police if two adjacent houses were broken into on the same night.

1. Given a list of non-negative integers representing the amount of money of each house, determine the maximum amount of money you can rob tonight without alerting the police.
2. What if all houses are arranged in a circle?

大概意思就是街上有一排房子，有个房子里有一定数量的钱，如果小偷在同一天晚上偷了两座相邻的房子，就会触发警报系统，第一问就是问在小偷不触发警报系统的前提下，怎样偷到的钱最多。

动态规划问题最重要的就是求出递归式，把原问题的最优化化为子问题的最优化。对这个问题来说，唯一的约束条件是不能同时抢劫相邻的两座房子，假设共有n座房子，分两种情况讨论：

(1). 如果你抢劫了第n座房子，很明显你能得到第n座房子的钱，但你只能抢劫第n-2座房子了，因为你不想被抓起来；

(2). 此时不选择抢劫第n座房子，当然待会你就可以抢劫第n-1座房子了

但这两种选择哪种是最优的了，很明显取决与这两个子问题的最优解，所以我们把原问题的最优解化成求解子问题的最优解，根据上述两种情况划分，可以很容易的写出递推公式：

$$dp(n)=MAX\{dp(n-1),dp(n-2)+money(n)\}$$

其中money(n)表示抢劫第n座房子得到的钱(注意在写程序是money[n-1],因为数组下标从0开始)，根据递推公式，采用自底向上的方法，就能计算出最优的结果。

伪代码如下所示：
![](http://i.imgur.com/MW11e6T.png)

再来看第二问，第二问的意思就是如果房子不是排成一排而是围成一圈，该怎样才能偷到最多的钱？
我们随机的从一圈n座房子中选中一座房子，现在我们面临两个选择，抢还是不抢？

(1) 假设我们选择抢劫这座房子，当然我们能得到这座房子的钱，但是为了避免触发警报我们不能抢劫它周围的两座房子了，那么剩下的n-3座房子就没有构成一个圈了，那么问题也就规约成第一问的情况了；
(2) 假如我们没有抢劫这座房子，那么去掉这座房子，剩下的n-1座房子不是就不构成圈了吗，所以还是回到第一问的问题。

经过分析，所以我们得到递推公式为:

$$dp_{circle(n)}=MAX\{dp(n-1),dp(n-3)+money(n)\}$$


# 问题举例：Maximum profit of transactions

问题如下：

Say you have an array for which the i-th element is the price of a given stock on day i.
Design an algorithm and implement it to find the maximum profit. You may complete at most two transactions.

Note: You may not engage in multiple transactions at the same time (ie,you must sell the stock before you buy again).

这个题目的意思是给你每天股票的价格，你能最多进行两次交易(一次交易包括买进和卖出)，怎样才能获得最大的收益。

我们先来讨论只在一次交易的情况下，怎么求得收益最大化，其实这个问题就是给你一个数组，求出这组数里面不是[逆序数](http://buptldy.github.io/2016/01/06/%E5%88%86%E6%B2%BB%E7%AD%96%E7%95%A5%5BDivide%20and%20Conquer%5D/)(因为卖出肯定在买进之后)但相差最大的两个数的差,比如5，1，3，2，4中，以价钱1买进，价钱4卖出可以获得最大的收益3。

我们用动态规划法来分析这个问题，通过自底向上的方法，分析如何从前i天的最大收益推出前i+1天的最大收益，已知前i天的最大收益和前i天的最低价格：

1. 第i+1天的价格大于minPrice（已遍历数据的最低价），此时只要对max(i)（前i天的最大获益）和prices[i + 1] - minPrice（第i+1天卖出所能得到的获益）取大值就能得出max(i + 1)
2. 第i+1天的价格小于等于minPrice，那么在第i+1天卖出所得到的获益必然是小于max(i)（这里哪怕考虑极端情况：给出的数据是整体递减的，那么最佳的卖出时机也是当天买当天卖，获益为0，所以不会存在获益是负值的情况），所以max(i + 1) = max(i)。而且，对于之后的数据而言，minPrice需要更新了，因为对于之后的数据，在第i+1天买进必然比在0到i天之间的任何时候买进的获益都要多（因为第i+1天是0到i+1区间内的最低价）。

所以通过上述动态规划的方法可以求出只进行一次交易的最大收益，但我们题目中问的是最多进行两次交易的情况下，我们可以把Prices[] 分成两部分Prices[0...m] 和 Prices[m...length]  ，分别计算在这两部分内做交易的最大收益，方法就是上面所说的一次交易的方法，第一步扫描，先计算出子序列[0,...,i]中的最大利润，用一个数组保存下来，时间是O(n)。 第二步是逆向扫描，计算子序列[i,...,n-1]上的最大利润，这一步同时就能结合上一步的结果计算最终的最大利润了，这一步也是O(n)。 所以最后算法的复杂度就是O(n)。

[算法代码下载(CPP)](https://github.com/BUPTLdy/Algorithms/tree/master/Maximum%20profit%20of%20transactions)


# 参考

Best Time to Buy and Sell Stock I II III IV@LeetCode：[http://segmentfault.com/a/1190000002565570](http://segmentfault.com/a/1190000002565570 "Best Time to Buy and Sell Stock I II III IV@LeetCode")
LeetCode-Best Time to Buy and Sell Stock系列：[http://www.tuicool.com/articles/rMJZj2](http://www.tuicool.com/articles/rMJZj2 "LeetCode-Best Time to Buy and Sell Stock系列")
