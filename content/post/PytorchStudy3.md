---
title: "Pytorch实践（刘二大人）3：反向传播"
tags:
- 机器学习
- 代码实践
categories: 机器学习
image: post/img/15.jpg
---

## 基础知识

### PyTorch语法

[PyTorch语法](https://xn--8mr985eba830aiye.vip/p/pytorch语法/)

### 张量（Tensor）

是一个广义的数学概念

* **0阶张量**：标量（Scalar），如温度、质量
* **1阶张量**：向量（vector），如速度、力
* **2阶张量**：矩阵（Matrix），如应力张量、图像像素矩阵
* **更高阶张量**：如RGB图像（3阶张量：$高度\times 宽度\times 通道$[^1]）、视频数据（4阶张量：$时间\times 高度\times 宽度\times 通道$）[^1]

### 计算图（Computational Graph）

是一种用图形化方式表示数学运算流程的工具。它将复杂的计算过程分解为**节点（操作或变量）**和**边（数据流动）**，直观展示数据如何通过一系列运算得到最终结果。计算图是深度学习框架（如TensorFlow、PyTorch）实现自动微分和反向传播的核心基础。



## 思路

1. **前向传播**：计算预测值和损失
2. **反向传播**：计算梯度
3. **更新参数**：梯度下降
4. **验证更新后的模型**
5. **下一次反向传播**



## 一个具体的例子

#### 设定初始条件

* **训练样本**：$x=2,y_{pred}=4$（真实模型是$y=2x$，即$w=2$）
* **初始参数**：$w=1.0$（随机初始化）
* **学习率**：$\eta=0.1$



#### 前向传播

1. **计算预测值**：$y_{pred}=w\cdot x=2.0$
2. **计算损失**：$L=(y_{pred} - y_{true})^2=4.0$



#### 反向传播

1. **前向传播**：
   1. **计算**$\frac{\partial L}{\partial y_{pred}}$：$\frac{\partial L}{\partial y_{pred}}=2(y_{pred}-y_{true})=2(2.0-4.0)=-4.0$
   2. **计算**$\frac {\partial y_{pred}}{\partial w}$：$\frac {\partial y_{pred}}{\partial w}=x=2.0$

2. **链式法则组合梯度**：$\frac{\partial L}{\partial w}=\frac{\partial L}{\partial y_{pred}}\cdot \frac {\partial y_pred}{\partial w}=(-4.0)\times 2.0=-8.0$
3. **更新参数**：$w_{new}=w_{old}-\eta \frac{\partial L}{\partial w}=1.0-0.1\times (-8.0)=1.8$
4. **验证更新后的模型**：
   1. **预测值**：$y_{pred}=1.8\times 2=3.6$
   2. **损失**：$L=(3.6-4.0)^2=0.16$



#### 下一次反向传播

1. **计算梯度**：$\frac{\partial L}{\partial y_{pred}}=2(3.6-4.0)=-0.8$，$\frac {\partial y_{pred}}{\partial w}=2.0$，$\frac{\partial L}{\partial w}=(-0.8)\times 2.0=-1.6$
2. **更新参数**：$w=1.8-0.1\times(-1.6)=1.96$
3. **新损失**：$y_{pred}=1.96\times 2=3.92$，$L=(3.92-4.0)^2=0.0064$

## 代码实现

```python
import torch

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = torch.tensor([1.0])  # w的初值为1.0
w.requires_grad = True  # 需要计算梯度


def forward(x):
    return x * w  # w是一个Tensor


def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2


print("predict (before training)", 4, forward(4).item())

for epoch in range(100):
    for x, y in zip(x_data, y_data):
        l = loss(x, y)
        l.backward()
        print('\tgrad:', x, y, w.grad.item())
        w.data = w.data - 0.01 * w.grad.data

        w.grad.data.zero_()

    print('progress:', epoch, l.item())  

print("predict (after training)", 4, forward(4).item())
```



[^1]: 每个通道都是一个二维矩阵，组合成三维张量