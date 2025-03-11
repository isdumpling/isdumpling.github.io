---
title: 机器学习（李宏毅）笔记 1：预测本频道观测人数（上）
date: 2025-02-25T17:21:44+08:00
params:
  author: 一只饺子
tags:
  - 机器学习
categories: 机器学习
image: post/img/4.jpg
---

* 机器学习约等于寻找一个函数。比如：
	* **speech recognition:**  输入一段“How are you”的语音，我们得到$f(音频) = "How  are you"$
	* **image recognition**: 输入一张猫的图片，我们得到$f(cat.jpg)= "cat"$

* 函数的不同类型有：
	* **regression**：回归函数。该函数输出一个标量。比如，当我们预测第二天的$PM2.5$，我们输入：$PM2.5today, temperature, Concentration of O_3$，经过一个函数得到$PM2.5 of tomorrow$
	* **Classification**: 分类函数。给定一些选项，该函数会输出正确的选项
	* **structure learning**：结构学习。让机器学会创造这件事情



### 如何使用机器寻找一个函数？



总的来说分为三步:
1. 定义带函数的参数
2. 定义损失函数
3. 优化参数

这里引入了一个例子。一youtuber根据自己从前每日视频播放量，预测第二天的视频播放量



#### 1. 定义带参数的函数

我们有$y=f(数据集)->y=b+wx_1$

$w$和$b$是未知参数，对于初始$w_0$和$b_0$，我们通过~~猜~~<mark>domain knowledge</mark>（专业领域知识）来进行推测



#### 2. 定义损失函数

* 损失函数是一个带参数的函数：$L(b, w)$
* 损失函数能评测一组数据的优劣如何

我们假设$L(0.5k, 1)$，则$y =  b  +  wx_1 - >y=0.5k+1x_1$
该youtuber的2017/01/01的播放量是$4.8k$，01/02的播放量为$4.9k$，01/03的播放量为$7.5k$

带入01/01的数据，我们预测01/02为$y=0.5+1\times 4.8=5.3$

那么我们可得:

* $e=|y-\widehat{y}|$，$L$是<mark>平均绝对误差</mark>(absolute error, MAE)，且有$MAE = \frac{1}{N}\sum_{i= 1}^{N}|y_{pred}^{(i)}-y_{true}^{(i)}|$

  

* $e=(y-\widehat y)^2$，$L$是<mark>均方误差</mark>(mean square error, MSE)，且有$MSE=\frac{1}{N}\sum_{i=1}^N(y_{pred}^{(i)}-y_{true}^{(i)})^2$



$y=b+\sum_i c_isigmoid(b_i+\sum_j w_{ij}x_j)$
