---
layout:     post
title:      "10 Minutes to pandas"
subtitle:   "10 Minutes to pandas"
date:       2016-03-25 10:00:00
author:     "Ldy"
header-img: "img/tag-bg1.jpg"
tags:
    - Python
    - pandas
---


# 10分钟简单介绍pandas

首先，导入模块如下所示：


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```

# pandas数据结构：Series
<!--more-->
Series可以简单地被认为是一维的数组。 Series 和一维数组最主要的区别在于 Series类型具有索引( index ),可以和另一个编程中常见的数据结构哈希( Hash )联系起来。

创建Series类型数据结构，如果没有传入索引，pandas默认的索引为从0开始的整数。


```python
s = pd.Series([1,3,5,np.nan,6,8])
```


```python
s
```




    0     1
    1     3
    2     5
    3   NaN
    4     6
    5     8
    dtype: float64



# pandas数据结构：DataFrame

DataFrame 是将数个 Series 按列合并而成的二维数据结构,每一列单独取出来是一个 Series ,这和 SQL 数据库中取出的数据是很类似的。所以,按
列对一个 DataFrame 进行处理更为方便,用户在编程时注意培养按列构建数据的思维。 DataFrame 的优势在于可以方便地处理不同类型的列,因此,就不要考虑如何对一个全是浮点数的 DataFrame 求逆之类的问题了,处理这种问题还是把数据存成 NumPy 的 matrix 类型比较便利一些。


通过传入 numpy array数据创建 DataFrame：


```python
dates = pd.date_range('20130101', periods=6)
```


```python
dates
```




    DatetimeIndex(['2013-01-01', '2013-01-02', '2013-01-03', '2013-01-04',
                   '2013-01-05', '2013-01-06'],
                  dtype='datetime64[ns]', freq='D')




```python
df = pd.DataFrame(np.random.randn(6,4), index=dates, columns=list('ABCD'))
```


```python
df
```
<table border="1" class="dataframe">   <thead>     <tr style="text-align: right;">       <th></th>       <th>A</th>       <th>B</th>       <th>C</th>       <th>D</th>     </tr>   </thead>   <tbody>     <tr>       <th>2013-01-01</th>       <td>0.212880</td>       <td>0.351725</td>       <td>-1.350579</td>       <td>-0.107403</td>     </tr>     <tr>       <th>2013-01-02</th>       <td>-0.857903</td>       <td>-1.783324</td>       <td>1.162888</td>       <td>-0.488226</td>     </tr>     <tr>       <th>2013-01-03</th>       <td>-0.245746</td>       <td>-0.226585</td>       <td>1.749624</td>       <td>1.140817</td>     </tr>     <tr>       <th>2013-01-04</th>       <td>0.032400</td>       <td>-0.264382</td>       <td>0.125095</td>       <td>-1.322739</td>     </tr>     <tr>       <th>2013-01-05</th>       <td>-2.260707</td>       <td>0.064878</td>       <td>0.231025</td>       <td>0.682991</td>     </tr>     <tr>       <th>2013-01-06</th>       <td>0.603739</td>       <td>1.490709</td>       <td>0.249649</td>       <td>1.822501</td>     </tr>   </tbody> </table>  

传入字典对象创建DataFrame：


```python
df2 = pd.DataFrame({ 'A' : 1.,
....:                'B' : pd.Timestamp('20130102'),
....:                'C' : pd.Series(1,index=list(range(4)),dtype='float32'),
....:                'D' : np.array([3] * 4,dtype='int32'),
....:                'E' : pd.Categorical(["test","train","test","train"]),
....:                'F' : 'foo' })
```


```python
df2
```





<table border="1" class="dataframe">   <thead>     <tr style="text-align: right;">       <th></th>       <th>A</th>       <th>B</th>       <th>C</th>       <th>D</th>       <th>E</th>       <th>F</th>     </tr>   </thead>   <tbody>     <tr>       <th>0</th>       <td>1</td>       <td>2013-01-02</td>       <td>1</td>       <td>3</td>       <td>test</td>       <td>foo</td>     </tr>     <tr>       <th>1</th>       <td>1</td>       <td>2013-01-02</td>       <td>1</td>       <td>3</td>       <td>train</td>       <td>foo</td>     </tr>     <tr>       <th>2</th>       <td>1</td>       <td>2013-01-02</td>       <td>1</td>       <td>3</td>       <td>test</td>       <td>foo</td>     </tr>     <tr>       <th>3</th>       <td>1</td>       <td>2013-01-02</td>       <td>1</td>       <td>3</td>       <td>train</td>       <td>foo</td>     </tr>   </tbody> </table>    

```python
df2.F
```




    0    foo
    1    foo
    2    foo
    3    foo
    Name: F, dtype: object




```python
df2.A
```




    0    1
    1    1
    2    1
    3    1
    Name: A, dtype: float64



查看数据顶部或底部的几行：


```python
df.head(2)
```





<table border="1" class="dataframe">   <thead>     <tr style="text-align: right;">       <th></th>       <th>A</th>       <th>B</th>       <th>C</th>       <th>D</th>     </tr>   </thead>   <tbody>     <tr>       <th>2013-01-01</th>       <td>0.212880</td>       <td>0.351725</td>       <td>-1.350579</td>       <td>-0.107403</td>     </tr>     <tr>       <th>2013-01-02</th>       <td>-0.857903</td>       <td>-1.783324</td>       <td>1.162888</td>       <td>-0.488226</td>     </tr>   </tbody> </table>  



```python
df.tail(3)
```





<table border="1" class="dataframe">   <thead>     <tr style="text-align: right;">       <th></th>       <th>A</th>       <th>B</th>       <th>C</th>       <th>D</th>     </tr>   </thead>   <tbody>     <tr>       <th>2013-01-04</th>       <td>0.032400</td>       <td>-0.264382</td>       <td>0.125095</td>       <td>-1.322739</td>     </tr>     <tr>       <th>2013-01-05</th>       <td>-2.260707</td>       <td>0.064878</td>       <td>0.231025</td>       <td>0.682991</td>     </tr>     <tr>       <th>2013-01-06</th>       <td>0.603739</td>       <td>1.490709</td>       <td>0.249649</td>       <td>1.822501</td>     </tr>   </tbody> </table>   

显示行列索引和里面的值;


```python
df.index
```




    DatetimeIndex(['2013-01-01', '2013-01-02', '2013-01-03', '2013-01-04',
                   '2013-01-05', '2013-01-06'],
                  dtype='datetime64[ns]', freq='D')




```python
df.columns
```




    Index([u'A', u'B', u'C', u'D'], dtype='object')




```python
df.values
```




    array([[ 0.21287973,  0.35172526, -1.35057903, -0.10740265],
           [-0.85790301, -1.78332415,  1.16288782, -0.48822551],
           [-0.24574644, -0.22658458,  1.74962416,  1.14081656],
           [ 0.03240016, -0.26438175,  0.12509531, -1.32273918],
           [-2.26070679,  0.06487812,  0.23102475,  0.68299111],
           [ 0.60373902,  1.4907093 ,  0.24964875,  1.82250141]])



显示数据的简单统计：


```python
df.describe()
```





<table border="1" class="dataframe">   <thead>     <tr style="text-align: right;">       <th></th>       <th>A</th>       <th>B</th>       <th>C</th>       <th>D</th>     </tr>   </thead>   <tbody>     <tr>       <th>count</th>       <td>6.000000</td>       <td>6.000000</td>       <td>6.000000</td>       <td>6.000000</td>     </tr>     <tr>       <th>mean</th>       <td>-0.419223</td>       <td>-0.061163</td>       <td>0.361284</td>       <td>0.287990</td>     </tr>     <tr>       <th>std</th>       <td>1.026018</td>       <td>1.061053</td>       <td>1.056953</td>       <td>1.148160</td>     </tr>     <tr>       <th>min</th>       <td>-2.260707</td>       <td>-1.783324</td>       <td>-1.350579</td>       <td>-1.322739</td>     </tr>     <tr>       <th>25%</th>       <td>-0.704864</td>       <td>-0.254932</td>       <td>0.151578</td>       <td>-0.393020</td>     </tr>     <tr>       <th>50%</th>       <td>-0.106673</td>       <td>-0.080853</td>       <td>0.240337</td>       <td>0.287794</td>     </tr>     <tr>       <th>75%</th>       <td>0.167760</td>       <td>0.280013</td>       <td>0.934578</td>       <td>1.026360</td>     </tr>     <tr>       <th>max</th>       <td>0.603739</td>       <td>1.490709</td>       <td>1.749624</td>       <td>1.822501</td>     </tr>   </tbody> </table>   

数据转置：


```python
df.T
```





<table border="1" class="dataframe">   <thead>     <tr style="text-align: right;">       <th></th>       <th>2013-01-01 00:00:00</th>       <th>2013-01-02 00:00:00</th>       <th>2013-01-03 00:00:00</th>       <th>2013-01-04 00:00:00</th>       <th>2013-01-05 00:00:00</th>       <th>2013-01-06 00:00:00</th>     </tr>   </thead>   <tbody>     <tr>       <th>A</th>       <td>0.212880</td>       <td>-0.857903</td>       <td>-0.245746</td>       <td>0.032400</td>       <td>-2.260707</td>       <td>0.603739</td>     </tr>     <tr>       <th>B</th>       <td>0.351725</td>       <td>-1.783324</td>       <td>-0.226585</td>       <td>-0.264382</td>       <td>0.064878</td>       <td>1.490709</td>     </tr>     <tr>       <th>C</th>       <td>-1.350579</td>       <td>1.162888</td>       <td>1.749624</td>       <td>0.125095</td>       <td>0.231025</td>       <td>0.249649</td>     </tr>     <tr>       <th>D</th>       <td>-0.107403</td>       <td>-0.488226</td>       <td>1.140817</td>       <td>-1.322739</td>       <td>0.682991</td>       <td>1.822501</td>     </tr>   </tbody> </table>



按某个索引排序：


```python
df.sort_index(axis=1,ascending=False)
```





<table border="1" class="dataframe">   <thead>     <tr style="text-align: right;">       <th></th>       <th>D</th>       <th>C</th>       <th>B</th>       <th>A</th>     </tr>   </thead>   <tbody>     <tr>       <th>2013-01-01</th>       <td>-0.107403</td>       <td>-1.350579</td>       <td>0.351725</td>       <td>0.212880</td>     </tr>     <tr>       <th>2013-01-02</th>       <td>-0.488226</td>       <td>1.162888</td>       <td>-1.783324</td>       <td>-0.857903</td>     </tr>     <tr>       <th>2013-01-03</th>       <td>1.140817</td>       <td>1.749624</td>       <td>-0.226585</td>       <td>-0.245746</td>     </tr>     <tr>       <th>2013-01-04</th>       <td>-1.322739</td>       <td>0.125095</td>       <td>-0.264382</td>       <td>0.032400</td>     </tr>     <tr>       <th>2013-01-05</th>       <td>0.682991</td>       <td>0.231025</td>       <td>0.064878</td>       <td>-2.260707</td>     </tr>     <tr>       <th>2013-01-06</th>       <td>1.822501</td>       <td>0.249649</td>       <td>1.490709</td>       <td>0.603739</td>     </tr>   </tbody> </table>  


按数据的值排序：


```python
df.sort_values(by='B')
```





<table border="1" class="dataframe">   <thead>     <tr style="text-align: right;">       <th></th>       <th>A</th>       <th>B</th>       <th>C</th>       <th>D</th>     </tr>   </thead>   <tbody>     <tr>       <th>2013-01-02</th>       <td>-0.857903</td>       <td>-1.783324</td>       <td>1.162888</td>       <td>-0.488226</td>     </tr>     <tr>       <th>2013-01-04</th>       <td>0.032400</td>       <td>-0.264382</td>       <td>0.125095</td>       <td>-1.322739</td>     </tr>     <tr>       <th>2013-01-03</th>       <td>-0.245746</td>       <td>-0.226585</td>       <td>1.749624</td>       <td>1.140817</td>     </tr>     <tr>       <th>2013-01-05</th>       <td>-2.260707</td>       <td>0.064878</td>       <td>0.231025</td>       <td>0.682991</td>     </tr>     <tr>       <th>2013-01-01</th>       <td>0.212880</td>       <td>0.351725</td>       <td>-1.350579</td>       <td>-0.107403</td>     </tr>     <tr>       <th>2013-01-06</th>       <td>0.603739</td>       <td>1.490709</td>       <td>0.249649</td>       <td>1.822501</td>     </tr>   </tbody> </table>



选出某一类：(同df.A)


```python
df['A']
```




    2013-01-01    0.212880
    2013-01-02   -0.857903
    2013-01-03   -0.245746
    2013-01-04    0.032400
    2013-01-05   -2.260707
    2013-01-06    0.603739
    Freq: D, Name: A, dtype: float64



通过[]切分出几行：


```python
df[0:3]
```





<table border="1" class="dataframe">   <thead>     <tr style="text-align: right;">       <th></th>       <th>A</th>       <th>B</th>       <th>C</th>       <th>D</th>     </tr>   </thead>   <tbody>     <tr>       <th>2013-01-01</th>       <td>0.212880</td>       <td>0.351725</td>       <td>-1.350579</td>       <td>-0.107403</td>     </tr>     <tr>       <th>2013-01-02</th>       <td>-0.857903</td>       <td>-1.783324</td>       <td>1.162888</td>       <td>-0.488226</td>     </tr>     <tr>       <th>2013-01-03</th>       <td>-0.245746</td>       <td>-0.226585</td>       <td>1.749624</td>       <td>1.140817</td>     </tr>   </tbody> </table>




      df['20130102':'20130104']


<table border="1" class="dataframe">   <thead>     <tr style="text-align: right;">       <th></th>       <th>A</th>       <th>B</th>       <th>C</th>       <th>D</th>     </tr>   </thead>   <tbody>     <tr>       <th>2013-01-02</th>       <td>-0.857903</td>       <td>-1.783324</td>       <td>1.162888</td>       <td>-0.488226</td>     </tr>     <tr>       <th>2013-01-03</th>       <td>-0.245746</td>       <td>-0.226585</td>       <td>1.749624</td>       <td>1.140817</td>     </tr>     <tr>       <th>2013-01-04</th>       <td>0.032400</td>       <td>-0.264382</td>       <td>0.125095</td>       <td>-1.322739</td>     </tr>   </tbody> </table>




通过标签选择：



```python
df.loc[dates[0],['A','B']]
```



```python
A    0.212880
B    0.351725
Name: 2013-01-01 00:00:00, dtype: float64
```


通过位置选取：




```python
df.iloc[1:3,0:2]
```





<table border="1" class="dataframe">   <thead>     <tr style="text-align: right;">       <th></th>       <th>A</th>       <th>B</th>     </tr>   </thead>   <tbody>     <tr>       <th>2013-01-02</th>       <td>-0.857903</td>       <td>-1.783324</td>     </tr>     <tr>       <th>2013-01-03</th>       <td>-0.245746</td>       <td>-0.226585</td>     </tr>   </tbody> </table>  


reindex方法，能够增加行和列：


```python
df1 = df.reindex(index=dates[0:4], columns=list(df.columns) + ['E'])
```


```python
df1.loc[dates[0]:dates[1],'E'] = 1
df1
```





<table border="1" class="dataframe">   <thead>     <tr style="text-align: right;">       <th></th>       <th>A</th>       <th>B</th>       <th>C</th>       <th>D</th>       <th>E</th>     </tr>   </thead>   <tbody>     <tr>       <th>2013-01-01</th>       <td>0.212880</td>       <td>0.351725</td>       <td>-1.350579</td>       <td>-0.107403</td>       <td>1</td>     </tr>     <tr>       <th>2013-01-02</th>       <td>-0.857903</td>       <td>-1.783324</td>       <td>1.162888</td>       <td>-0.488226</td>       <td>1</td>     </tr>     <tr>       <th>2013-01-03</th>       <td>-0.245746</td>       <td>-0.226585</td>       <td>1.749624</td>       <td>1.140817</td>       <td>NaN</td>     </tr>     <tr>       <th>2013-01-04</th>       <td>0.032400</td>       <td>-0.264382</td>       <td>0.125095</td>       <td>-1.322739</td>       <td>NaN</td>     </tr>   </tbody> </table>



丢失数据的处理:

去掉有丢失数据的所有行：


```python
df1.dropna(how='any')
```





<table border="1" class="dataframe">   <thead>     <tr style="text-align: right;">       <th></th>       <th>A</th>       <th>B</th>       <th>C</th>       <th>D</th>       <th>E</th>     </tr>   </thead>   <tbody>     <tr>       <th>2013-01-01</th>       <td>0.212880</td>       <td>0.351725</td>       <td>-1.350579</td>       <td>-0.107403</td>       <td>1</td>     </tr>     <tr>       <th>2013-01-02</th>       <td>-0.857903</td>       <td>-1.783324</td>       <td>1.162888</td>       <td>-0.488226</td>       <td>1</td>     </tr>   </tbody> </table>  


填充丢失数据


```python
df1.fillna(value=5)
```





<table border="1" class="dataframe">   <thead>     <tr style="text-align: right;">       <th></th>       <th>A</th>       <th>B</th>       <th>C</th>       <th>D</th>       <th>E</th>     </tr>   </thead>   <tbody>     <tr>       <th>2013-01-01</th>       <td>0.212880</td>       <td>0.351725</td>       <td>-1.350579</td>       <td>-0.107403</td>       <td>1</td>     </tr>     <tr>       <th>2013-01-02</th>       <td>-0.857903</td>       <td>-1.783324</td>       <td>1.162888</td>       <td>-0.488226</td>       <td>1</td>     </tr>     <tr>       <th>2013-01-03</th>       <td>-0.245746</td>       <td>-0.226585</td>       <td>1.749624</td>       <td>1.140817</td>       <td>5</td>     </tr>     <tr>       <th>2013-01-04</th>       <td>0.032400</td>       <td>-0.264382</td>       <td>0.125095</td>       <td>-1.322739</td>       <td>5</td>     </tr>   </tbody> </table>  


判断是否有丢失数据：


```python
 pd.isnull(df1)
```





<table border="1" class="dataframe">   <thead>     <tr style="text-align: right;">       <th></th>       <th>A</th>       <th>B</th>       <th>C</th>       <th>D</th>       <th>E</th>     </tr>   </thead>   <tbody>     <tr>       <th>2013-01-01</th>       <td>False</td>       <td>False</td>       <td>False</td>       <td>False</td>       <td>False</td>     </tr>     <tr>       <th>2013-01-02</th>       <td>False</td>       <td>False</td>       <td>False</td>       <td>False</td>       <td>False</td>     </tr>     <tr>       <th>2013-01-03</th>       <td>False</td>       <td>False</td>       <td>False</td>       <td>False</td>       <td>True</td>     </tr>     <tr>       <th>2013-01-04</th>       <td>False</td>       <td>False</td>       <td>False</td>       <td>False</td>       <td>True</td>     </tr>   </tbody> </table>

# 读取文件

写csv文件：


```python
df.to_csv('foo.csv')
```

读csv文件：


```python
pd.read_csv('foo.csv')
```





<table border="1" class="dataframe">   <thead>     <tr style="text-align: right;">       <th></th>       <th>Unnamed: 0</th>       <th>A</th>       <th>B</th>       <th>C</th>       <th>D</th>     </tr>   </thead>   <tbody>     <tr>       <th>0</th>       <td>2013-01-01</td>       <td>0.212880</td>       <td>0.351725</td>       <td>-1.350579</td>       <td>-0.107403</td>     </tr>     <tr>       <th>1</th>       <td>2013-01-02</td>       <td>-0.857903</td>       <td>-1.783324</td>       <td>1.162888</td>       <td>-0.488226</td>     </tr>     <tr>       <th>2</th>       <td>2013-01-03</td>       <td>-0.245746</td>       <td>-0.226585</td>       <td>1.749624</td>       <td>1.140817</td>     </tr>     <tr>       <th>3</th>       <td>2013-01-04</td>       <td>0.032400</td>       <td>-0.264382</td>       <td>0.125095</td>       <td>-1.322739</td>     </tr>     <tr>       <th>4</th>       <td>2013-01-05</td>       <td>-2.260707</td>       <td>0.064878</td>       <td>0.231025</td>       <td>0.682991</td>     </tr>     <tr>       <th>5</th>       <td>2013-01-06</td>       <td>0.603739</td>       <td>1.490709</td>       <td>0.249649</td>       <td>1.822501</td>     </tr>   </tbody> </table>  



```python

```

# 参考资料

[10 Minutes to pandas¶
](http://pandas.pydata.org/pandas-docs/stable/10min.html)
