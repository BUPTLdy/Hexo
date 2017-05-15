---
layout:     post
title:      "Python Binary Search Tree implementation"
subtitle:   "Python实现二叉搜索树"
date:       2016-05-09 10:00:00
author:     "Ldy"
header-img: "img/tag-bg1.jpg"
tags:
    - Data Structure
    - Python
---


<center>
![](http://7xritj.com1.z0.glb.clouddn.com/16-5-9/90589523.jpg)
</center>
<!--more-->


二叉查找树（英语：Binary Search Tree），也称二叉搜索树、有序二叉树（英语：ordered binary tree），排序二叉树（英语：sorted binary tree），是指一棵空树或者具有下列性质的二叉树：

- 任意节点的左子树不空，则左子树上所有结点的值均小于它的根结点的值；
- 任意节点的右子树不空，则右子树上所有结点的值均大于它的根结点的值；
- 任意节点的左、右子树也分别为二叉查找树；
- 没有键值相等的节点。
如下所示为一棵二叉查找树：
<center>
![](http://7xritj.com1.z0.glb.clouddn.com/16-5-8/57493358.jpg)
</center>

# 定义节点类

二叉树的每个节点有三个属性:
- 左节点
- 右节点
- 节点值

所以用Python定义一个节点类为：
```python
class Node:
    def __init__(self, data,left=None,right=None):
        self.left = left
        self.right = right
        self.data = data
```

现在来创建一个根节点为8的树：
```python
root=Node(8)
```
如下图所示：
<center>
![](http://7xritj.com1.z0.glb.clouddn.com/16-5-8/11936419.jpg)
</center>

# 插入节点

比较要插入数据和根节点的大小，递归的调用插入方法
```python
class Node:
    ...
    def insert(self, data):
        if self.data:#如果存在根节点
            if data < self.data:
                if self.left is None:
                    self.left = Node(data)
                else:
                    self.left.insert(data)
            elif data > self.data:
                if self.right is None:
                    self.right = Node(data)
                else:
                    self.right.insert(data)
        else:
            self.data = data
```

现在来插入三个节点：

```python
root.insert(3)
root.insert(10)
root.insert(1)
```
现在的二叉树如下所示：
<center>
![](http://7xritj.com1.z0.glb.clouddn.com/16-5-8/48415300.jpg)
</center>

继续增加一些节点，让二叉树看起来更完整：
```python
root.insert(6)
root.insert(4)
root.insert(7)
root.insert(14)
root.insert(13)
```
<center>
![](http://7xritj.com1.z0.glb.clouddn.com/16-5-9/52855138.jpg)
</center>

# 二叉查找树的查找

```python
class Node:
    ...
    def lookup(self, data, parent=None):
        if data < self.data:
            if self.left is None:
                return None, None
            return self.left.lookup(data, self)
        elif data > self.data:
            if self.right is None:
                return None, None
            return self.right.lookup(data, self)
        else:
            return self, parent
```

查找是否存在节点6，并返回这个节点和其父节点：
```python
node, parent = root.lookup(6)
```
其中查找的过程如下所示：
<center>
![](http://7xritj.com1.z0.glb.clouddn.com/16-5-9/13534624.jpg)
</center>

# 删除节点

在删除节点时，首先得统计节点孩子的个数：

```python
class Node:
    ...
    def children_count(self):
        cnt = 0
        if self.left:
            cnt += 1
        if self.right:
            cnt += 1
        return cnt
```

删除节点，分三种情况：
- 要删除的节点没有孩子节点
- 要删除的节点有一个孩子节点
- 要删除的节点有两个孩子节点

```python
class Node:
    ...
    def delete(self, data):
        node, parent = self.lookup(data)
        if node is not None:
            children_count = node.children_count()
                if children_count == 0:
                    # if node has no children, just remove it
                    if parent:
                        if parent.left is node:
                            parent.left = None
                        else:
                            parent.right = None
                        del node
                    else:
                        self.data = None
                elif children_count == 1:
                      # if node has 1 child
                      # replace node with its child
                    if node.left:
                        n = node.left
                    else:
                        n = node.right
                    if parent:
                        if parent.left is node:
                            parent.left = n
                        else:
                            parent.right = n
                        del node
                    else:
                        self.left = n.left
                        self.right = n.right
                        self.data = n.data
                else:
                    # if node has 2 children
                    # find its successor
                    parent = node
                    successor = node.right
                    while successor.left:
                        parent = successor
                        successor = successor.left
                    # replace node data by its successor data
                    node.data = successor.data
                    # fix successor's parent's child
                    if parent.left == successor:
                        parent.left = successor.right
                    else:
                        parent.right = successor.right
```

# 打印二叉树

按照中序打印二叉树，前序和后序只需要修改打印的顺序就行。
```python
class Node:
    ...
    def print_tree(self):
        """
        Print tree content inorder
        """
        if self.left:
            self.left.print_tree()
        print self.data,
        if self.right:
            self.right.print_tree()
```

按层次打印一个树：
```python
class Node:
    ...
    def print_each_level(self):
      # Start off with root node
      thislevel = [self]

      # While there is another level
      while thislevel:
        nextlevel = list()
        #Print all the nodes in the current level, and   store the next level in a list
        for node in thislevel:
          print node.data
          if node.left: nextlevel.append(node.left)
          if node.right: nextlevel.append(node.right)
          print
          thislevel = nextlevel


```
# 比较两棵树

```python
class Node:
    ...
    def compare_trees(self, node):
        if node is None:
            return False
        if self.data != node.data:
            return False
        res = True
        if self.left is None:
            if node.left:
                return False
        else:
            res = self.left.compare_trees(node.left)
        if res is False:
            return False
        if self.right is None:
            if node.right:
                return False
        else:
            res = self.right.compare_trees(node.right)
        return res
```

# 二叉树的重建

根据前序遍历和中序遍历来重建树，重建的原理可以参看这篇博文[根据二叉树的前序和中序求后序](http://blog.csdn.net/hinyunsin/article/details/6315502):
```python
def rebuilt(preorder,inorder):
    if preorder=='' or inorder=='':
        return None
    root=preorder[0]
    index=inorder.index(root)
    return Node(root,
                rebuilt(preorder[1:1+index],inorder[0:index]),
                rebuilt(preorder[index+1:],inorder[index+1:]))
```

根据中序和后序来重建树：

```python
def rebuilt1(inorder,postorder):
    if postorder=='' or inorder=='':
        return None
    root=postorder[-1]
    index=inorder.index(root)
    return Node(root,
                rebuilt1(inorder[0:index],postorder[0:index]),
                rebuilt1(inorder[index+1:],postorder[index:-1]))

```


# 参考
[二叉搜索树](https://zh.wikipedia.org/wiki/%E4%BA%8C%E5%85%83%E6%90%9C%E5%B0%8B%E6%A8%B9)
[Binary Search Tree library in Python](http://www.laurentluce.com/posts/binary-search-tree-library-in-python/)
