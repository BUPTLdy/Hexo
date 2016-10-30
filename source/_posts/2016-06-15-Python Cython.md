---
layout:     post
title:      "Speed-up with Cython and Numpy in Python"
subtitle:   "Python速度优化-Cython中numpy以及多线程的使用"
date:       2016-06-15 10:00:00
author:     "Ldy"
header-img: "img/tag-bg1.jpg"
tags:
    - Python
    - Cython
---

<center>
![](http://7xritj.com1.z0.glb.clouddn.com/16-6-15/20612896.jpg)
</center>
<!--more-->

## Cython代码和Python代码区别

代码运行在[IPython-Notebook](https://ipython.org/notebook.html)中，在IPython-Notebook中导入cython环境。

```python
%load_ext cython
```

Cython可以在Python中掺杂C和C++的静态类型，cython编译器可以把Cython源码编译成C或C++代码，编译后的代码可以单独执行或者作为Python中的模型使用。Cython中的强大之处在于可以把Python和C结合起来，它使得看起来像Python语言的Cython代码有着和C相似的运行速度。

我们使用一个简单的Fibonacci函数来比较下Python和Cython的区别：


```python
#python
def fib1(n):
    a,b=0.0,1.0
    for i in range(n):
        a,b=a+b,a
    return a
```

下面代码使用`%%cython`标志表示下面的代码使用cython编译
```python
%%cython

def fib2(int n):
    cdef double a=0.0, b=1.0
    for i in range(n):
        a,b = a+b,a
    return a
```

通过比较上面的代码，为了把Python中的动态类型转换为Cython中的静态类型，我们用`cdef`来定义C语言中的变量`i`，`a`，`b`。
我们用C语言实现Fibonacci函数，然后通过Cython用Python封装，其中`cfib.h`为Fibonacci函数C语言实现，如下：
```c
double cfib(int n) {
  int i;
  double a=0.0, b=1.0, tmp;
  for (i=0; i<n; ++i) {
    tmp = a; a = a + b; b = tmp;
  }
  return a;
}

```


```python
%%cython

cdef extern from "/home/ldy/MEGA/python/cython/cfib.h":
    double cfib(int n)  
def fib3(n):
    """Returns the nth Fibonacci number."""
    return cfib(n)
```

比较不同方法的运行时间：


```python

%timeit result=fib1(1000)

%timeit result=fib2(1000)

%timeit result=fib3(1000)

```

    10000 loops, best of 3: 73.6 µs per loop
    1000000 loops, best of 3: 1.94 µs per loop
    1000000 loops, best of 3: 1.92 µs per loop

## Cython代码的编译

Cython代码的编译为Python可调用模块的过程主要分为两步：第一步是cython编译器把Cython代码优化成C或C++代码；第二步是使用C或C++编译器编译产生的C或C++代码得到Python可调用的模块。

我们通过一个`setup.py`脚本来编译上面写的`fib.pyx`Cython代码，如下所示，关键就在第三行，`cythonize`函数的作用是通过cython编译器把Cython代码转换为C代码，`setup`函数则是把产生的C代码转换成Python可调用模块。


```python
from distutils.core import setup
from Cython.Build import cythonize
setup(ext_modules=cythonize('fib.pyx'))
#setup(ext_modules=cythonize('*.pyx','fib1.pyx'))也可以一次编译多个Cython文件
```

写好`setup.py`文件后，就可以通过下述命令执行编译：
```shell
python setup.py build_ext --inplace
```
执行后产生了`fib.c`代码以及`fib.so`文件，以及一些中间结果保存在build文件夹里。


```python
import os
os.chdir('/home/ldy/MEGA/python/cython/test')
os.getcwd()
!ls
```

    build  fib.c  fib.pyx  fib.so  setup.py


通过Python调用产出的`fib.so`模块：


```python
import fib
fib.fib2(90)

```




    2.880067194370816e+18




## Cython中类型的定义

为什么Cython和Python比会提高很多性能，主要原因有两点：一是Python是解释型语言，在运行之前Python解释器把Python代码解释成Python字节码运行在Python虚拟机上，Python虚拟机把Python字节码最终翻译成CPU能执行的机器码；而Cython代码是事先直接编译成可被Python调用的机器码，在运行时可直接执行。第二个主要的原因是Python是动态类型，Python解释器在解释时需要判断类型，然后再提取出底层能够运行的数据以及操作；然而C语言等比较底层的语言是静态类型，编译器直接提取数据进行操作产生机器码。

Cython中使用`cdef`来定义静态类型：
```
cdef int i
cdef int j
cdef float f
```
也可以一次定义多个：
```
cdef:
    int i
    int j
    float f
```

Cython中还允许在静态类型和动态类型同时存在及相互赋值：


```python
%%cython
cdef int a=1,b=2,c=3
list_of_ints=[a,b,c]
list_of_ints.append(4)
a=list_of_ints[1]
print a,list_of_ints
```

    2 [1, 2, 3, 4]


声明Python类型为静态类型，Cython支持把一些Python内置的如`list`,`tuple`,`dict`等类型声明为静态类型，这样声明使得它们能像正常Python类型一样使用，但是需要约束成只能是他们所申明的类型，不能随意变动。


```python
%%cython
cdef:
    list names
    dict name_num

name_num={'jerry':1,'Tom':2,'Bell':3}
names=list(name_num.keys())
print names
other_names=names#动态类型可以从静态类型的Python对象初始化
del other_names[0]#因为引用了同一个list，所以都会删除第一个元素
print names,other_names
other_names=tuple(other_names)#names和other_names的区别在于names只能是list类型，
print other_names           #other_names可以引用任何类型

```

    ['Bell', 'jerry', 'Tom']
    ['jerry', 'Tom'] ['jerry', 'Tom']
    ('jerry', 'Tom')


## Cython中numpy的使用

我们先构造一个函数来测试下使用纯Python时的运算时间来做对比，这个函数的作用是对一副输入图像求梯度（不必过分关注函数的功能，在这只是使用这个函数作为测试）。函数的输入数据是`indata`一个像素为1400\*1600的图片；输出为`outdata`,为每个像素梯度值，下面是这个函数的纯Python实现：


```python
import numpy as np
indata = np.random.rand(1400,1600)
outdata = np.zeros(shape=indata.shape, dtype='float64')  # eventually holds our output
from numpy.lib import pad
print("shape before", indata.shape)
indata = pad(indata, (1, 1), 'reflect', reflect_type='odd')  # allow edge calcs
print("shape after", indata.shape)

import math
def slope(indata, outdata):
    I = outdata.shape[0]
    J = outdata.shape[1]
    for i in range(I):
        for j in range(J):
            # percent slope using Zevenbergen-Thorne method
            # assume edges added, inarr is offset by one on both axes cmp to outarr
            dzdx = (indata[i+1, j] - indata[i+1, j+2]) / 2  # assume cellsize == one unit, otherwise (2 * cellsize)
            dzdy = (indata[i, j+1] - indata[i+2, j+1]) / 2
            slp = math.sqrt((dzdx * dzdx) + (dzdy * dzdy)) * 100  # percent slope (take math.atan to get angle)
            outdata[i, j] = slp
```

    ('shape before', (1400, 1600))
    ('shape after', (1402, 1602))


测试运行时间，为5.31 s每个循环


```python
%timeit slope(indata, outdata)
```

    1 loop, best of 3: 5.31 s per loop


重置输出：


```python
def reset_outdata():
    outdata = np.zeros(shape=indata.shape, dtype='float64')

reset_outdata()
```

使用Cython重写求图像梯度函数,其中函数`slope_cython2`使用Cython里的numpy类型，并重写了里面的开方函数，其中`%%cython -a`表示使用cython编译Cython代码，并可以对照显示编译器把Cython代码编译成的C代码。


```python
%%cython
import cython
cimport numpy as np
ctypedef np.float64_t DTYPE_t
@cython.boundscheck(False)
def slope_cython2(np.ndarray[DTYPE_t, ndim=2] indata, np.ndarray[DTYPE_t, ndim=2] outdata):
    cdef int I, J
    cdef int i, j, x
    cdef double k, slp, dzdx, dzdy
    I = outdata.shape[0]
    J = outdata.shape[1]
    for i in range(I):
        for j in range(J):
            dzdx = (indata[i+1, j] - indata[i+1, j+2]) / 2
            dzdy = (indata[i, j+1] - indata[i+2, j+1]) / 2
            k = (dzdx * dzdx) + (dzdy * dzdy)
            slp = k**0.5 * 100
            outdata[i, j] = slp
```

测试运行时间：208ms,快了有25倍左右


```python
%timeit slope_cython2(indata, outdata)
```

    1 loop, best of 3: 208 ms per loop

## Cython中多进程

Cython还支持[并行运算](http://docs.cython.org/src/userguide/parallelism.html),后台由OpenMP支持，所以在编译Cython语言时需要加上如下代码第一行所示的标记。在进行并行计算时，需使用`nogil`关键词来释放Python里的[GIL](https://wiki.python.org/moin/GlobalInterpreterLock)锁,当代码中只有C而没有Python对象时，这样做是安全的。


```python
%%cython --compile-args=-fopenmp --link-args=-fopenmp --force

import cython
from cython.parallel import prange, parallel

@cython.boundscheck(False)
def slope_cython_openmp(double [:, :] indata, double [:, :] outdata):
    cdef int I, J
    cdef int i, j, x
    cdef double k, slp, dzdx, dzdy
    I = outdata.shape[0]
    J = outdata.shape[1]
    with nogil, parallel(num_threads=4):
        for i in prange(I, schedule='dynamic'):
            for j in range(J):
                dzdx = (indata[i+1, j] - indata[i+1, j+2]) / 2
                dzdy = (indata[i, j+1] - indata[i+2, j+1]) / 2
                k = (dzdx * dzdx) + (dzdy * dzdy)
                slp = k**0.5 * 100
                outdata[i, j] = slp
```


```python
reset_outdata()
%timeit slope_cython_openmp(indata, outdata)
```

    10 loops, best of 3: 78.2 ms per loop


测试的时间如上所示，多进程大概快了2.7倍左右。
