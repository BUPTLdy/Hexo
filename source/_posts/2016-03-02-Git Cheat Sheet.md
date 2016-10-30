---
layout:     post
title:      "Git Cheat Sheet"
subtitle:   "GIT CHEAT SHEET"
date:       2016-03-02 10:00:00
author:     "Ldy"
header-img: "img/tag-bg1.jpg"
tags:
    - Git
    - Linux
---

1. 创建版本库

	初始化一个Git仓库，使用git init命令。

	添加文件到Git仓库，分两步：

	第一步，使用命令git add <file>，注意，可反复多次使用，添加多个文件；
<!--more-->
	第二步，使用命令git commit，完成。

	要随时掌握工作区的状态，使用git status命令。

	如果git status告诉你有文件被修改过，用git diff可以查看修改内容。

2. 版本回退

	HEAD指向的版本就是当前版本，因此，Git允许我们在版本的历史之间穿梭，使用命令git reset --hard commit_id。Git必须知道当前版本是哪个版本，在Git中，用HEAD表示当前版本，上一个版本就是HEAD^，上上一个版本就是HEAD^^，当然往上100个版本写100个^比较容易数不过来，所以写成HEAD~100。

	穿梭前，用git log可以查看提交历史，以便确定要回退到哪个版本。

	要重返未来，用git reflog查看命令历史，以便确定要回到未来的哪个版本。

3. 工作区和暂存区

	工作区（Working Directory）：就是你在电脑里能看到的目录。

	版本库（Repository）：工作区有一个隐藏目录.git，这个不算工作区，而是Git的版本库。

	Git的版本库里存了很多东西，其中最重要的就是称为stage（或者叫index）的暂存区，还有Git为我们自动创建的第一个分支master，以及指向master的一个指针叫HEAD。

<center>
![](http://7xritj.com1.z0.glb.clouddn.com/16-3-10/74272820.jpg)
</center>

	前面讲了我们把文件往Git版本库里添加的时候，是分两步执行的：

	第一步是用git add把文件添加进去，实际上就是把文件修改添加到暂存区；

	第二步是用git commit提交更改，实际上就是把暂存区的所有内容提交到当前分支。

	因为我们创建Git版本库时，Git自动为我们创建了唯一一个master分支，所以，现在，git commit就是往master分支上提交更改。

	你可以简单理解为，需要提交的文件修改通通放到暂存区，然后，一次性提交暂存区的所有修改。

	Git是如何跟踪修改的:每次修改，如果不add到暂存区，那就不会加入到commit中。

	场景1：当你改乱了工作区某个文件的内容，想直接丢弃工作区的修改时，用命令git checkout -- file。git checkout其实是用版本库里的版本替换工作区的版本，无论工作区是修改还是删除，都可以“一键还原”。

	场景2：当你不但改乱了工作区某个文件的内容，还添加到了暂存区时，想丢弃修改，分两步，第一步用命令git reset HEAD file，就回到了场景1，第二步按场景1操作。

	场景3：已经提交了不合适的修改到版本库时，想要撤销本次提交，参考版本回退一节，不过前提是没有推送到远程库。

4. 远程仓库

	要关联一个远程库，使用命令git remote add origin git@server-name:path/repo-name.git；

	关联后，使用命令git push -u origin master第一次推送master分支的所有内容；

	此后，每次本地提交后，只要有必要，就可以使用命令git push origin master推送最新修改；

5. 创建和合并分支

	查看分支：git branch

	创建分支：git branch <name>

	切换分支：git checkout <name>

	创建+切换分支：git checkout -b <name>

	合并某分支到当前分支：git merge <name>

	删除分支：git branch -d <name>

	当Git无法自动合并分支时，就必须首先解决冲突。解决冲突后，再提交，合并完成。

	用git log --graph命令可以看到分支合并图。

	修复bug时，我们会通过创建新的bug分支进行修复，然后合并，最后删除；

	当手头工作没有完成时，先把工作现场git stash一下，然后去修复bug，修复后，再git stash pop，回到工作现场。

	开发一个新feature，最好新建一个分支；

	如果要丢弃一个没有被合并过的分支，可以通过git branch -D <name>强行删除。

	查看远程库信息，使用git remote -v；

	本地新建的分支如果不推送到远程，对其他人就是不可见的；

	从本地推送分支，使用git push origin branch-name，如果推送失败，先用git pull抓取远程的新提交；

	在本地创建和远程分支对应的分支，使用git checkout -b branch-name origin/branch-name，本地和远程分支的名称最好一致；

	建立本地分支和远程分支的关联，使用git branch --set-upstream branch-name origin/branch-name；

	从远程抓取分支，使用git pull，如果有冲突，要先处理冲突。

	命令git tag <name>用于新建一个标签，默认为HEAD，也可以指定一个commit id；

6. 标签管理

	git tag -a <tagname> -m "blablabla..."可以指定标签信息；

	git tag -s <tagname> -m "blablabla..."可以用PGP签名标签；

	命令git tag可以查看所有标签。

	命令git push origin <tagname>可以推送一个本地标签；

	命令git push origin --tags可以推送全部未推送过的本地标签；

	命令git tag -d <tagname>可以删除一个本地标签；

	命令git push origin :refs/tags/<tagname>可以删除一个远程标签

7. Github使用

	在GitHub上，可以任意Fork开源仓库；

	自己拥有Fork后的仓库的读写权限；

	可以推送pull request给官方仓库来贡献代码。

	忽略某些文件时，需要编写.gitignore；

	.gitignore文件本身要放到版本库里，并且可以对.gitignore做版本管理！



**参考资料**

![](http://7xritj.com1.z0.glb.clouddn.com/16-3-10/93928228.jpg)

[Git教程-廖雪峰的官方网站](http://www.liaoxuefeng.com/wiki/0013739516305929606dd18361248578c67b8067c8c017b000)
