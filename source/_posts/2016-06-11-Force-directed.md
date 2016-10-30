---
layout:     post
title:      "Force-Directed Graph Visualization Based in Location"
subtitle:   "基于位置约束的图可视化"
date:       2016-06-11 10:00:00
author:     "Ldy"
header-img: "img/tag-bg1.jpg"
tags:
    - D3
    - Data Visualization
---

<center>
![](http://7xritj.com1.z0.glb.clouddn.com/16-6-11/65263838.jpg)
</center>
<!--more-->

## 任务介绍

图是表现社交网络、知识图谱等关系的主要形式，对图的节点进行布局是图可视化的重要内容。然而，现有方法大多在布局时没有考虑节点地理位置对布局的约束。比如在POI点评应用中，我们希望一个“餐厅”节点出现在它实际的地理位置上，或者在热点事件应用中，希望“北京”节点出现在“上海”节点的北方（上方）。在布局中加入地理位置约束，能够使图的可视化结果更好的与位置关联，包含地理信息相关隐喻，在增加其承载信息量的同时，更好的辅助地理空间数据的可视分析。
任务1：调研图可视化中节点布局相关方法，特别是力引导方法和二分图布局方法，形成小综述；
任务2：将二分图中一类节点加入绝对地理位置或彼此间相对位置不变作为约束条件，改进一种基于力引导布局的二分图可视化方法，给出模型、公式、算法流程描述；
任务3：基于给定数据集（两类节点，一类节点包含地理坐标），选择一种可视化工具（如VTK、D3等），对上述改进算法进行实现。

数据集形式如下所示：

文件PlaceTolation.txt内容如下，分别为地名和经纬度
```
地名	纬度,经度
北京	39.90,116.40
北京市	39.90,116.40
北京站	39.90,116.40
北京路	39.90,116.40
天安门	39.90,116.38
崇文	39.88,116.43
崇文区	39.88,116.43
......
```

文件TitlePlace.txt内容如下,分别为序号,新闻标题和从该新闻中抽取出来的地名实体
```
1	落马高官忏悔：从未感觉到还有党组织存在	中国
2	佩帅：442阵型没问题对方进球很无解不怪门将	利物浦,切尔西
3	今日数据趣谈：半场20+命中率8成5小加变大加	北京,德安,奎尔,孟菲斯
4	工业领域控煤计划将出台：2020年力争节煤1.6亿吨	北京,河北,山西
5	公交乘客与司机扭打发生车祸致1人重伤(图)	呼和浩特,呼和浩特市,青城,内蒙古,赛罕区,青洲
6	深圳机场行人围观飞机起降被撞倒已致5死24伤	深圳
7	云南临沧发生3.5级地震震源深度14千米	中国,云南省,临沧市,沧源佤族自治县,云南
......
```

需要构建的二分图中两类节点分别为新闻标题和地名，节点间的关系为标题和地名的映射关系（多对多的），其中地名节点具有经纬度属性。

## 数据清洗

从数据中可以看出，有很多地名是重复的，比如北京其实和北京市是同一个意思，还有什么天安门，崇文区都是属于北京的，从经纬度上来看，应该把他们都归为一类，不然在地图上也不好显示，都是相聚很短的重合的点，基于以上考虑，我们可以根据经纬度把每个地点替换为其的所属的省或直辖市的名称。

要想判读每个地名所属的省市，那我们就需要每个省市的经纬度范围，在网上找到的中国地图的[JSON](http://www.ourd3js.com/demo/rm/R-10.0/china.geojson)文件,其中包含了每个省边界的经纬度值，为一系列的点，判断某个地点属于哪一个省实际上就是根据地点的经纬度判断这一点是否在某所有省边界点围成的多边形里，也就是一个[Point in Polygon](https://en.wikipedia.org/wiki/Point_in_polygon)问题。

Python matplotlib包中的Path提供了相应的函数：
```python
import matplotlib.path as mplPath
bbPath = mplPath.Path(np.array([[0,0],[1,0],[1,1],[0,1]]))
bbPath.contains_point((0.5, 0.5))
```

## 力导向图的制作

力导向图中每一个节点都受到力的作用而运动，这种是一种非常绚丽的图表。

![](http://7xritj.com1.z0.glb.clouddn.com/16-5-22/7813444.jpg)

力导向图（Force-Directed Graph），是绘图的一种算法。在二维或三维空间里配置节点，节点之间用线连接，称为连线。各连线的长度几乎相等，且尽可能不相交。节点和连线都被施加了力的作用，力是根据节点和连线的相对位置计算的。根据力的作用，来计算节点和连线的运动轨迹，并不断降低它们的能量，最终达到一种能量很低的安定状态。力导向图能表示节点之间的多对多的关系。

d3.layout.force()包含了力导向算法的实现，其主要参数为：

d3.layout.force - 使用物理模拟排放链接节点的位置。
force.alpha - 取得或者设置力布局的冷却参数。
force.chargeDistance - 取得或者设置最大电荷距离。
force.charge - 取得或者设置电荷强度。
force.drag - 给节点绑定拖动行为。
force.friction - 取得或者设置摩擦系数。
force.gravity - 取得或者设置重力强度。
force.linkDistance - 取得或者设置链接距离。
force.linkStrength - 取得或者设置链接强度。
force.links - 取得或者设置节点间的链接数组。
force.nodes - 取得或者设置布局的节点数组。
force.on - 监听在计算布局位置时的更新。
force.resume - 重新加热冷却参数，并重启模拟。
force.size - 取得或者设置布局大小。
force.start - 当节点变化时启动或者重启模拟。
force.stop - 立即停止模拟。
force.theta - 取得或者设置电荷作用的精度。
force.tick - 运行布局模拟的一步。

关于d3.layout.force()的使用可参考[力导向图的制作](http://www.ourd3js.com/wordpress/?p=196)

## 具体实现

结合我们题目的实际要求，我们有两类节点：一类是地点节点，其位置要求固定；一类是新闻节点，其位置根据力导向算法计算得到，所以节点定义如下。
```javascript
var nodes = [
              {name:"青海",x:青海[0],y:青海[1],fixed:true,"group":1},
              {name:"河南",x:河南[0],y:河南[1],fixed:true,"group":1},
              {name:"山东",x:山东[0],y:山东[1],fixed:true,"group":1},
              .
              .
              .

              {name:"从WCBA争冠到无缘新赛季浙江女篮怎么了",fixed:false,"group":2},
              {name:"成都的哥:专车司机玩着跑半个月超过我月收入",fixed:false,"group":2},
              {name:"部分农村教师月薪不到2千暑假当小工补贴家用",fixed:false,"group":2}
                ];
```
其中第一类节点为固定地点节点，第二类节点为新闻节点，使用力导向算法计算节点的位置。所以我们需要提供地点节点的位置，在定义节点之前，加上地点经纬度：
```javascript
var 青海 =[96.5122866869,35.12781926];
var 河南 =[114.130772484,34.00715756];
var 山东 =[118.354817653,36.2612648184];
.
.
.
```

接下来是连线之间的定义，某一新闻里包含哪几个地点，则这几个地点就和这个新闻之间连一条线，其中0表示上面定义的第一个节点,185表示第186个节点。
```javascript
var edges = [
                {source:0,target:185},
                {source:0,target:204},
                {source:0,target:389},
                {source:0,target:430},
                {source:0,target:494},
                {source:1,target:42},
                .
                .
                .
              ]
```

定义好数据之后，就可以开始布局了

定义一个力导向图的布局如下。

```javascript
var force = d3.layout.force()
      .nodes(nodes) //指定节点数组
      .links(edges) //指定连线数组
      .size([width,height]) //指定作用域范围
      .linkDistance(150) //指定连线长度
      .charge([-400]); //相互之间的作用力
```
然后，使力学作用生效：

```javascript
force.start();    //开始作用
```

## 可视化

力学作业生效以后，新闻节点的坐标地址就会产生，根据产生的新闻坐标地址就可以绘制出整个可视化图。

分别绘制三种图形元素：

- line，线段，表示连线。
- circle，圆，表示节点。
- text，文字，描述节点。

代码如下：

```javascript
//添加连线
 var svg_edges = svg.selectAll("line")
     .data(edges)
     .enter()
     .append("line")
     .style("stroke","#ccc")
     .style("stroke-width",1);

 var color = d3.scale.category20();

 //添加节点
 var svg_nodes = svg.selectAll("circle")
     .data(nodes)
     .enter()
     .append("circle")
     .attr("r",20)
     .style("fill",function(d,i){
         return color(i);
     })
     .call(force.drag);  //使得节点能够拖动

 //添加描述节点的文字
 var svg_texts = svg.selectAll("text")
     .data(nodes)
     .enter()
     .append("text")
     .style("fill", "black")
     .attr("dx", 20)
     .attr("dy", 8)
     .text(function(d){
        return d.name;
     });
```
调用 call( force.drag ) 后节点可被拖动。force.drag() 是一个函数，将其作为 call() 的参数，相当于将当前选择的元素传到 force.drag() 函数中。

## 结果展示

可视化结果如下所示，在线演示地址:[http://buptldy.github.io/DEMO/news_map.html](http://buptldy.github.io/DEMO/news_map.html)

![](http://7xritj.com1.z0.glb.clouddn.com/16-5-21/3836519.jpg)
