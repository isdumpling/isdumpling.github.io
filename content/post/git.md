---
title: git
date: 2025-02-25T16:21:44+08:00
params:
  author: 一只饺子
tags:
- git
---

## 1. 与GitHub仓库建立连接并提交修改

1. SSH密钥配备完成且有相关权限
2. 初始化本地仓库：`git init`
3. 关联远程仓库：
	1. HTTPS: `git remote add origin https://github.com/username/repositoryname.git`
	2. SSH:`git remote add origin git@github.com:username/repositoryname.git`
4. 首次拉取文件：`git pull origin main`
5. 更改文件
6. 提交更改到本地：
	1. 添加所有更改：`git add .`
	2. 添加提交说明：`git commit -m "你的提交说明"`
	3. 推送到远程仓库：`git push -u origin main`

## 2. 更改分支

1. 查看本地分支（当前分支会用`*`标出）：`git branch`
2. 查看所有分支（本地+远程跟踪分支）：`git branch -a`
3. 切换现有分支
	1. 直接切换：`git checkout 分支名`。如：切换到`develop`分支：`git checkout develop`
	2. 拉取远程分支并切换：`git checkout -b 本地分支名 origin/远程分支名`。如：同步远程的`feature/login`分支到本地：`git checkout -b feature/login origin/feature/login`
4. 创建并切换到新分支：
	1. 创建新分支并立即切换：`git checkout -b 新分支名`
5. 推送新分支到远程仓库：`git push orign 分支名`