---
layout:     post
title:      "Implementing a Singly Linked List in Python"
subtitle:   "Python实现单链表"
date:       2016-05-09 11:00:00
author:     "Ldy"
header-img: "img/tag-bg1.jpg"
tags:
    - Data Structure
    - Python
---


<center>
![](http://7xritj.com1.z0.glb.clouddn.com/16-5-9/62221771.jpg)
</center>
<!--more-->



链表中最简单的一种是单向链表，它包含两个域，一个信息域和一个指针域。这个链接指向列表中的下一个节点，而最后一个节点则指向一个空值。一个单向链表的节点被分成两个部分。第一个部分保存或者显示关于节点的信息，第二个部分存储下一个节点的地址。单向链表只可向一个方向遍历。

<center>
![](http://7xritj.com1.z0.glb.clouddn.com/16-5-9/37737204.jpg)
</center>

# 链表节点类的实现

```python
class Node:
    def __init__(self,initdata):
        self.data = initdata
        self.next = None

    def getData(self):
        return self.data

    def getNext(self):
        return self.next

    def setData(self,newdata):
        self.data = newdata

    def setNext(self,newnext):
        self.next = newnext

```

生成一个节点对象：

```python
>>> temp = Node(93)
>>> temp.getData()
93
```
结构如下图所示：
<center>
![](http://7xritj.com1.z0.glb.clouddn.com/16-5-9/39620482.jpg)
</center>

# 链表类的实现

```python
class UnorderedList:

    def __init__(self):
        self.head = None
```

新建一个链表对象：
```python
>>> mylist = UnorderedList()
```

# 往链表前端中加入节点

```python
def add(self,item):
    temp = Node(item)
    temp.setNext(self.head)
    self.head = temp
```

```python
>>> mylist.add(31)
>>> mylist.add(77)
>>> mylist.add(17)
>>> mylist.add(93)
>>> mylist.add(26)
>>> mylist.add(54)
```

现在链表结构如下图所示：
<center>
![](http://7xritj.com1.z0.glb.clouddn.com/16-5-9/89034482.jpg)
</center>

# 在链表尾端添加节点

```python
def append(self,item):
    temp=Node(item)
    if self.head == None:
        self.head=item
    else:
        current=self.head
        while current.getNext()!=None:
            current=current.getNext
        current.setNext(temp)
```

# 链表的长度计算
```python
def size(self):
    count=0
    current=self.head
    while current.getNext !=None:
        count=count+1
        current=current.getNext

```
计算过程如下图所示：

<center>
![](http://7xritj.com1.z0.glb.clouddn.com/16-5-9/28270792.jpg)
</center>

# 寻找是否存在某一节点

```python
def serch(self,item):
    current=self.head
    while current.getNext()!=None:
        if current.getData==item:
            return True
        else:
            current=current.getNext()

    return False
```

# 删除某一节点

```python
def remove(self,item):
    current=self.head
    pre=None
    while current!=None:
        if current.getData()==item:
            if not pre:
                self.head=current.getNext()
            else:
                pre.setNext(current.getNext())
            break
        else:
            pre=current
            current=current.getNext()


```

# 链表反转

```python
def rev(self):
    pre=None
    current=self.head
    while current!=None:
        next=current.getNext()
        current.setNext=pre
        pre=current
        curren=next
    return pre    
```
# 链表成对调换

例如：`1->2->3->4`转换成`2->1->4->3`

```python
def pairswap(self):
    curren=self.head
    while curren!=None and curren.getNext().getNext()!=None:
        temp=curren.getData()
        curren.setData(curren.getNext().getData())
        curren.getNext().setData(temp)
        curren=curren.getNext().getNext()
```
