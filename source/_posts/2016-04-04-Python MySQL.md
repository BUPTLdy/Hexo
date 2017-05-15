---
layout:     post
title:      "Connect to MySQL in Python"
subtitle:   "Python连接MySQL"
date:       2016-04-04 11:00:00
author:     "Ldy"
header-img: "img/tag-bg1.jpg"
tags:
    - Python
    - MySQL
---

<center>
![](http://7xritj.com1.z0.glb.clouddn.com/16-4-4/94530193.jpg)
</center>

# 环境安装

  安装MySQL：


    sudo apt-get install mysql-server

<!--more-->
  安装MySQLdb模块：

    sudo apt-get install python-mysqldb

  测试是否安装成功：

    import MySQLdb

# MySQL建立数据库

  进入MySQL：

    mysql -u root -p

  进入MySQL并打开补全：

    mysql -u USER -p --local-infile=1  --auto-rehash

  建立一个数据库：

      create database testdb character set utf8;

  调用已经建立的数据库：

      use testdb;

  建立一个数据表：

      create table users
      (id int(2) not null primary key auto_increment,
      username varchar(40),
      password text,email text)
      default charset=utf8;


  显示表格：

        show tables;

  显示表格结构：

        desc users;

  表格中插入数据：

        insert into
        users(username,password,email)
        values("qiwsir","123123","qiwsir@gmail.com");

  查询表格内容：

        select * from users;

# Python操作数据库

  连接数据库：


    conn =
    MySQLdb.connect
    (host="localhost",user="root",
    passwd="123123",db="qiwsirtest",charset="utf8")

  用游标（指针）cursor的方式操作数据库：


    cur = conn.cursor()

  在表中插入一条记录：

    cur.execute("insert into
    users (username,password,email)
    values (%s,%s,%s)",("python","123456","python@gmail.com"))

  使插入的记录生效，提交：

    conn.commit()

  同时插入多条记录：


    cur.executemany("insert into
    users (username,password,email)
    values (%s,%s,%s)",
    (("google","111222","g@gmail.com"),
    ("facebook","222333","f@face.book"),
    ("github","333444","git@hub.com"),
    ("docker","444555","doc@ker.com")))

  要记得提交生效

  查询数据库：


    cur.execute("select * from users")

  上述操作只是得到结果的指针，想要显示查询结果，可以用到以下方法：

  - fetchall(self):接收全部的返回结果行.
  - fwetchmany(size=None):接收size条返回结果行.如果size的值大于返回的结果行的数量,则会返回cursor.arraysize条数据.
  - fetchone():返回一条结果行.
  - scroll(value,mode='relative'):移动指针到某一行.如果mode='relative',则表示从当前所在行移动value条,如果mode='absolute',则表示从结果集的第一行移动value条.


  python的MySQLdb指针提供了一个参数，可以实现将读取到的数据变成字典形式：

    cur = conn.cursor(cursorclass=MySQLdb.cursors.DictCursor)

  更新数据库：

    cur.execute("update users set username=%s where id=2",("mypython"))

   如果再下述连接数据库的语句中，如果没有指定具体的数据库，则连接到MySQL：

    conn = MySQLdb.connect
    (host="localhost",
    user="root",passwd="123123",
    db="qiwsirtest",charset="utf8")

  然后可以通过用conn.select_db()选择要操作的数据库：

    conn.select_db("testdb")


  不选数据库，而是要新建一个数据库，如下所示：

    cur = conn.cursor()
    cur.execute("create database newtest")

  建立数据库之后，就可以选择这个数据库，然后在这个数据库中建立一个数据表：

    cur.execute("create table newusers
    (id int(2) primary key auto_increment,
    username varchar(20), age int(2), email text)")

  当进行完有关数据操作之后，最后要做的就是关闭游标（指针）和连接。用如下命令实现：

    cur.close()
    conn.close()

# 参考


  [通过Python连接数据库](http://python.xiaoleilu.com/300/302.html)
