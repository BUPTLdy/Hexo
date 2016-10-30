---
layout:     post
title:      "Divide and Conquer"
subtitle:   "Karatsuba算法 逆序数统计"
date:       2016-01-06 10:00:00
author:     "Ldy"
header-img: "img/tag-bg1.jpg"
tags:
    - Algorithms
    - CPP
---


# 分治法简介：

使用分治法的前提是问题在结构上是**递归**的，为了解决这一问题，算法一次或多次递归地调用自身以解决紧密相关的若干个子问题。

分治模式在每层的递归时有三个步骤：
<!--more-->
	- **分解**原问题为若干子问题，这些子问题是原问题规模较小的实例。
	- **解决**这些子问题，递归的求解各子问题，当子问题规模最够小时，则直接求解。
	- **合并**这些子问题的解得到原始问题的解

# [Karatsuba](https://en.wikipedia.org/wiki/Karatsuba_algorithm) 算法

Karatsuba算法是一种快速乘法算法，在1960年由[Anatoly Karatsuba](https://en.wikipedia.org/wiki/Anatoly_Karatsuba)提出。普通乘法的复杂度是$O(n^2)$,而Karatsuba算法的时间复杂度为$O(n^{1.585})$。

现在运用分治的思想来阐述下Karatsuba算法的基本原理：

- 分解：原本要计算两个大数x，y的乘法,先把x,y分解成如下两部分，其中B为基。

$$x = x_1B^m + x_0$$

$$y = y_1B^m + y_0$$

其中 $x_0$ 和 $y_0$ 要小于 $B^m$。 现在$xy$可以写为:


\begin{equation}
\begin{cases}
xy = (x_1B^m + x_0)(y_1B^m + y_0)\\
xy = z_2B^{2m} + z_1B^m + z_0\\
z_2 = x_1y_1\\
z_1 = x_1y_0 + x_0y_1\\
z_0 = x_0y_0
\end{cases}
\end{equation}

原本上面的式子中一共需要计算4次乘法（与基的幂次相乘只要进行移位操作就行），但是注意到有：

$$z_1 = (x_1 + x_0)(y_1 + y_0) - z_2 - z_0$$

所以每次只需要计算三次乘法即可。

- 解决：通过上述分解步骤，我们每次还是需要3次乘法运算，然而这三次乘法运算我们可以继续递归的调用Karatsuba算法，直到数字足够小可以直接运用普通乘法求解。
- 合并：Karatsuba算法并没有涉及合并问题，通过公式$xy = z_2B^{2m} + z_1B^m + z_0$把最终结果求出来即可。

根据上述分析，当m=n/2时(n为乘数的长度)，递归的效率最高，所以递归公式为：

$$T(n) = 3 T(\lceil n/2\rceil) + cn + d$$

由递归公式根据[主定理](http://buptldy.github.io/2015/12/29/%E7%AE%97%E6%B3%95%E5%A4%8D%E6%9D%82%E5%BA%A6%E6%B8%90%E8%BF%9B%E6%A0%87%E5%8F%B7/)得到算法的复杂度为：

$$T(n) = \Theta(n^{\log_2 3})$$

Karatsuba算法伪代码：

```
		procedure karatsuba(num1, num2)
  			if (num1 < 10) or (num2 < 10)
				return num1*num2
  			/* calculates the size of the numbers */
  			m = max(size_base10(num1), size_base10(num2))
 			m2 = m/2
  			/* split the digit sequences about the middle */
 			high1, low1 = split_at(num1, m2)
  			high2, low2 = split_at(num2, m2)
  			/* 3 calls made to numbers approximately half the size */
  			z0 = karatsuba(low1,low2)
  			z1 = karatsuba((low1+high1),(low2+high2))
  			z2 = karatsuba(high1,high2)
  			return (z2*10^(2*m2))+((z1-z2-z0)*10^(m2))+(z0)
```
[Karatsuba算法代码下载(CPP)](https://github.com/BUPTLdy/Algorithms/tree/master/Karatsuba)

# 归并排序(MergeSort)

还是根据分治策略的思想，分为三个步骤：

- 分解：把要排序的n个元素序列分解成两个含有n/2个元素的子序列
<center>
![](http://i.imgur.com/umuDsOg.png)
</center>
- 解决：递归的调用分解，直到子序列能够直接排序
- 合并：归并排好序的子序列，直到得到原始问题的解

简单例子演示：

<center>
![](http://i.imgur.com/33u5yyl.gif)
</center>

# 逆序数统计(Counting Inversion)

问题定义：输入n个不同的数字$a_1,a_2,\cdots ,a_n$，计算有多少对逆序数，逆序数的定义为数字下标$i<j$但是数字$a_i>a_j$。比如数字序列2，4，1，3，5，数字2在数字1的前面，但比数字1大，所以是一对逆序数，上述数字序列共有3组逆序数：（2，1），（4，1），（4，3）。

统计逆序数要用到分治的思想，我们能很容易能想到先分解成两个子序列然后再递归求解每个子序列的逆序数。

现在主要的问题是解决，怎么计算两个不同子序列之间的逆序数，也就是怎么合并的问题，如下图所示，如果只是通过子序列之间每个数字的简单比较求解逆序数，则需要通过$n^2/4$次比较。所以时间复杂度为$T(n)=2T(n/2)+n^2/4=O(n^2)$(求解方法参考[主定理](http://buptldy.github.io/2015/12/29/%E7%AE%97%E6%B3%95%E5%A4%8D%E6%9D%82%E5%BA%A6%E6%B8%90%E8%BF%9B%E6%A0%87%E5%8F%B7/))

<center>
![](http://i.imgur.com/mDpKIUp.png)
</center>

所以这样和直接暴力解法相比，时间复杂度比没有降低，这时我们通过前面的归并排序想到，如果我们先对子序列进行排序(对子序列排序并不会影响两个子序列之间的逆序数对)，再统计子序列之间的逆序数。统计方法如下图所示，两个排好序的子序列之间进行逆序数统计，比如说第一个序列的第一个数字3比第二个序列的第一个数字2要大，则第一个序列3后面的数字肯定都要比2要大，所以直接统计出逆序数为6，根据这种思想可以把子序列之间的所有逆序数统计出来。


![](http://i.imgur.com/wJtHUMG.gif)

(gif generated by [ScreenToGif](https://screentogif.codeplex.com/))

上述合并求解逆序数的方法，先给子序列排序并同时计算逆序数，花费$O(n)$的时间，所以问题总的时间复杂度为：

$$T(n)=2T(n/2)+O(n)=O(nlogn)$$

[Counting Inversion算法代码下载(CPP)](https://github.com/BUPTLdy/Algorithms/tree/master/Counting%20Inversion)