---
layout:     post
title:      "Asymptotic Notation and Master theorem"
subtitle:   " \"Big O  Big Theta  Master theorem\""
date:       2015-12-29 10:00:00
author:     "Ldy"
header-img: "img/tag-bg1.jpg"
tags:
    - Algorithms
---

# 算法渐进记号
- Big $\Theta$ ：$\Theta$ 记号渐进的给出一个函数的上界和下届，表示同阶的函数簇。

- Big $O$：表示一个函数的渐进上界，用来限制算法的最坏情况运行时间。
<!--more-->
- Big $\Omega$：表示一个函数的渐进下界，算法运行的最好情况。

- Litter $o$: 和big $O$ 定义相似，区别主要是 big$O$ 提供的上界可能和函数是同阶的，litter $o$ 表示非渐进紧确的上界。

# 举例

<center>
![](http://i.imgur.com/GgsO2DX.png)
</center>

# 渐进记号与函数阶数的关系

其中$a，b$分别为函数$g(n)，f(n)$的阶数

<center>
![](http://i.imgur.com/wUDSXQf.png)
</center>
通常Big $\Theta$ 用来描述算法的最好和最坏的运行时间，Big $O$描述算法的最坏运行时间，Big $\Omega$描述算法的最好运行时间，经常使用的是Big $O$， 用来衡量算法的时间复杂度和空间复杂度。

#  主定理(master theorem)求解递归式

假设有递推关系式

$$T(n) = aT\left(\frac{n}{b}\right) + f(n)$$

其中$ a \geq 1 \mbox{, } b > 1$，n为问题规模，a为递推的子问题数量，n/b为每个子问题的规模（假设每个子问题的规模基本一样），f(n)为递推以外进行的计算工作，包含了问题分解和子问题合并的代价。

- 情况一
如果存在常数$\epsilon > 0$，有$f(n) = O\left( n^{\log_b (a) - \epsilon} \right)$，并且是多项式意义上的小于，那么

$$T(n) = \Theta\left( n^{\log_b a} \right)$$

- 情况二
如果存在常数k ≥ 0，有$f(n) = \Theta\left( n^{\log_b a} \log^{k} n \right)$那么

$$T(n) = \Theta\left( n^{\log_b a} \log^{k+1} n \right)$$

- 情况三
如果存在常数$\epsilon > 0$，有$f(n) = \Omega\left( n^{\log_b (a) + \epsilon} \right)$，并且是多项式意义上的大于，同时存在常数c < 1以及充分大的n，满足

$$a f\left( \frac{n}{b} \right) \le c f(n)$$

那么

$$T\left(n \right) = \Theta \left(f \left(n \right) \right)$$

简单举例：

$$T(n) = 9T\left(\frac{n}{3}\right) + n$$

对这个递归式，有a=9,b=3,f(n)=n,因此有$n^{log_b a}=n^2>n$,所以复杂度为：

$$T(n)=\Theta(n^2)$$
