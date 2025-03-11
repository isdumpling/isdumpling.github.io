---
title: "对于机器学习公式的例子"  # 标题含中文建议加双引号
author: "一只饺子"
summary: >  # 注意>符号后的换行
  本文通过线性回归的数学公式示例，演示机器学习算法的实现过程。
  包含梯度下降法的推导和Python代码实现，适合初学者理解基础原理。 
tags:  # 列表必须统一缩进2空格
  - 机器学习
  - 线性代数
categories:  # 分类建议使用层级结构
  - 机器学习基础
type: post
draft: false
math: true  # 如果包含公式需要启用数学支持
image: post/img/3.jpg
---



## 得到权重的值



以下是一个 **单层神经网络（感知机）** 的完整示例，通过 **手动模拟训练过程**，展示如何从数据中学习权重。我们以 **房价预测** 为例，假设数据仅包含一个样本，目标是让模型学会调整权重和偏置。



### **问题设定**
#### **输入特征**
- $ x_1 $（面积）：1（标准化后的值，如100平方米）
- $ x_2 $（房龄）：1（标准化后的值，如5年）

#### **真实输出**
- $ y_{\text{true}} = 3 $（单位：万元）

#### **模型结构**
- **线性模型**：$ y_{\text{pred}} = w_1 x_1 + w_2 x_2 + b $
- **初始参数**（随机初始化）：
  - 权重：$ w_1 = 0.5 $, $ w_2 = -0.3 $
  - 偏置：$ b = 0.2 $

#### **目标**
通过梯度下降，调整 $ w_1, w_2, b $，使得 $ y_{\text{pred}} $ 接近真实值 3。



### **训练过程**
#### **前向传播（计算预测值）**
$$
y_{\text{pred}} = w_1 x_1 + w_2 x_2 + b = 0.5 \times 1 + (-0.3) \times 1 + 0.2 = 0.5 - 0.3 + 0.2 = 0.4
$$
此时预测值为 0.4 万元，与真实值 3 相差较大。

#### **计算损失（均方误差）**
$$
\text{Loss} = (y_{\text{true}} - y_{\text{pred}})^2 = (3 - 0.4)^2 = 6.76
$$

#### **反向传播（计算梯度）**
对每个参数求偏导（链式法则）：
- **损失对 $ w_1 $ 的梯度**：
  $$
  \frac{\partial \text{Loss}}{\partial w_1} = 2(y_{\text{pred}} - y_{\text{true}}) \cdot x_1 = 2(0.4 - 3) \times 1 = -5.2
  $$
- **损失对 $ w_2 $ 的梯度**：
  $$
  \frac{\partial \text{Loss}}{\partial w_2} = 2(y_{\text{pred}} - y_{\text{true}}) \cdot x_2 = 2(0.4 - 3) \times 1 = -5.2
  $$
- **损失对 $ b $ 的梯度**：
  $$
  \frac{\partial \text{Loss}}{\partial b} = 2(y_{\text{pred}} - y_{\text{true}}) = 2(0.4 - 3) = -5.2
  $$

#### **更新参数（梯度下降）**
设定学习率 $ \eta = 0.1 $，更新规则：
$$
w_{\text{new}} = w_{\text{old}} - \eta \cdot \frac{\partial \text{Loss}}{\partial w}
$$
- **更新 $ w_1 $**：
  $$
  w_1 = 0.5 - 0.1 \times (-5.2) = 0.5 + 0.52 = 1.02
  $$
- **更新 $ w_2 $**：
  $$
  w_2 = -0.3 - 0.1 \times (-5.2) = -0.3 + 0.52 = 0.22
  $$
- **更新 $ b $**：
  $$
  b = 0.2 - 0.1 \times (-5.2) = 0.2 + 0.52 = 0.72
  $$



### **更新后的预测**
使用新参数重新计算预测值：
$$
y_{\text{pred}} = 1.02 \times 1 + 0.22 \times 1 + 0.72 = 1.02 + 0.22 + 0.72 = 1.96
$$
损失更新为：
$$
\text{Loss} = (3 - 1.96)^2 = 1.08
$$
**仅一次迭代，损失从 6.76 下降到 1.08**，说明权重调整有效。



### **多轮迭代后的结果**
重复上述过程（假设学习率不变）：

| 迭代次数 | $ w_1 $ | $ w_2 $ | $ b $  | $ y_{\text{pred}} $ | Loss   |
|----------|-----------|-----------|----------|-----------------------|--------|
| 0        | 0.5       | -0.3      | 0.2      | 0.4                  | 6.76   |
| 1        | 1.02      | 0.22      | 0.72     | 1.96                 | 1.08   |
| 2        | 1.45      | 0.65      | 1.17     | 2.60                 | 0.16   |
| 3        | 1.68      | 0.89      | 1.43     | 2.96                 | 0.0016 |

经过3次迭代，预测值 $ y_{\text{pred}} = 2.96 $ 接近真实值3，损失降至0.0016。



### **关键结论**
- **权重的本质**：模型通过梯度下降，沿着损失减小的方向调整权重，逐步逼近真实值。
- **学习率的作用**：学习率 $ \eta $ 控制参数更新步幅（过大可能导致震荡，过小收敛慢）。
- **实际训练**：真实场景中需使用大量数据分批训练，而非单个样本。



**附：Python代码模拟**

```python
# 初始参数
w1, w2, b = 0.5, -0.3, 0.2
x1, x2, y_true = 1, 1, 3
eta = 0.1

for epoch in range(3):
    # 前向传播
    y_pred = w1*x1 + w2*x2 + b
    loss = (y_true - y_pred)**2
  
    # 计算梯度
    dL_dw1 = 2*(y_pred - y_true)*x1
    dL_dw2 = 2*(y_pred - y_true)*x2
    dL_db = 2*(y_pred - y_true)
  
    # 更新参数
    w1 -= eta * dL_dw1
    w2 -= eta * dL_dw2
    b -= eta * dL_db
  
    print(f"Epoch {epoch}: w1={w1:.2f}, w2={w2:.2f}, b={b:.2f}, y_pred={y_pred:.2f}, Loss={loss:.4f}")
```



## $y = b + \sum_i c_i \, \text{sigmoid}(b_i + \sum_j w_{ij} x_j)$

假设我们要根据房屋的两个特征预测房价
* **特征1($x_1$)**：面积（平方米）
* **特征2($x_2$)**：房龄（年）

我们设计一个简单的神经网络，结构如下：
* **输入层：** 两个特征（$x_1,x_2$）
* **隐藏层**： 2个神经元（$i=1,2$）
* **输出层**： 1个输出（房价$y$）

### step 1：设定参数值

假设模型已经训练完成，参数如下：

**隐藏层参数**

| 神经元      | 权重$w_{i1}$（面积权重） | 权重$w_{i2}$(房龄权重) | 偏置$b_i$ |
| -------- | ---------------- | ---------------- | ------- |
| 1$(i=1)$ | 0.8              | -0.2             | 0.5     |
| 2$(i=2)$ | 0.5              | -0.6             | -0.3    |

**输出层参数**

| 权重$c_i$  | 偏置$b$ |
| -------- | ----- |
| $c_1=10$ | $b=5$ |
| $c_2=-8$ |       |


### step 2：输出数据

假设有一套房子的特征值为：
* 面积$x_1=100m^2$
* 房龄$x_2=5年$

### step 3：计算隐藏层输出

对每个隐藏层神经元，计算$z_i\,=\,b_i\,+\,w_{i1}x_1\,+\,w_{i2}x_2$，然后通过$sigmoid$激活函数得到$a_i=sigmoid(z_i)$

**神经元1($i=1$)的计算**

$$
\begin{aligned}
z_1 = b_1 + w_{11}x_1 + w_{12}x_2 = 0.5 + 0.8 \times 100 + (-0.2) \times 5 = 0.5 + 80 - 1 = 79.5\\
a_1 = \text{sigmoid}(79.5) = \frac{1}{1 + e^{-79.5}} \approx 1.0 \quad (\text{几乎完全激活})
\end{aligned}
$$

**神经元2($i=2$)的计算**

$$
\begin{aligned}
z_2 = b_2 + w_{21}x_1 + w_{22}x_2 = -0.3 + 0.5 \times 100 + (-0.6) \times 5 = -0.3 + 50 - 3 = 46.7\\
a_2 = \text{sigmoid}(46.7) = \frac{1}{1 + e^{-46.7}} \approx 1.0 \quad (\text{几乎完全激活})
\end{aligned}
$$

### step 4：计算输出层结果

$$
y = b + c_1 a_1 + c_2 a_2 = 5 + 10 \times 1.0 + (-8) \times 1.0 = 5 + 10 - 8 = 7
$$



## $L(\theta)\approx L(\theta ^{'})+L(\theta - \theta^{'})g+\frac{1}{2}(\theta-\theta^{'})^{T}H(\theta-\theta ^{'})$

由于~~线性代数学艺不精~~热爱线性代数，重新推导这个公式

1. **回忆一维泰勒展开**

例如，在$x'$附件展开$f(x)$
$$
f(x)\approx f(x')+f'(x')(x-x')+\frac{1}{2}f"(x')(x-x')^2
$$


对于泰勒展开公式：
$$
f(x_0,x)=\sum_{i=0}^n \frac{f^{(i)}(x_0)}{i!}(x-x_0)^i
$$


2. **扩展到多维情况（参数$\theta$是向量）**



$L(\theta)\approx L(\theta ^{'})+L(\theta - \theta^{'})g+\frac{1}{2}(\theta-\theta^{'})^{T}H(\theta-\theta ^{'})$



在多维情况下，参数是向量$\theta = [\theta_1,\theta_2,...\theta_n]^T$，梯度$g$是一阶导数的推广



对于**Hessian**矩阵，是多元函数的二阶偏导数构成的矩阵
$$
H = \nabla^2 f = \begin{bmatrix}
\frac{\partial^2 f}{\partial \theta_1^2} & \frac{\partial^2 f}{\partial \theta_1 \partial \theta_2} & \cdots & \frac{\partial^2 f}{\partial \theta_1 \partial \theta_n} \\
\frac{\partial^2 f}{\partial \theta_2 \partial \theta_1} & \frac{\partial^2 f}{\partial \theta_2^2} & \cdots & \frac{\partial^2 f}{\partial \theta_2 \partial \theta_n} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial^2 f}{\partial \theta_n \partial \theta_1} & \frac{\partial^2 f}{\partial \theta_n \partial \theta_2} & \cdots & \frac{\partial^2 f}{\partial \theta_n^2}
\end{bmatrix}
$$


**为什么需要转置 $(\theta - \theta')^\top$？**

- **维度匹配**：假设 $\theta$ 是 $n \times 1$ 向量，梯度 $g$ 也是 $n \times 1$，Hessian H$ $H$是 $n \times n$。  
  - 一阶项：$(\theta - \theta ')^Tg$是$n \times 1$向量，梯度$g$也是$n\times 1$，Hessian $H$是$n \times n$（标量）
    - 一阶项：$(\theta - \theta ')g$是$1\times n*n\times n * n \times 1=1\times 1$（标量）
- **数学必要性**：转置确保矩阵乘法维度相容。



3. **一个具体的例子**

##### **1. 定义函数**  
设损失函数 $L(\theta) = \theta_1^2 + 2\theta_2^2 + \theta_1\theta_2$，参考点 $\theta' = [0, 0]^\top$。

##### **2. 计算梯度 $g$**  
$$
g = \nabla L(\theta') = \begin{bmatrix} 2\theta_1 + \theta_2 \\ 4\theta_2 + \theta_1 \end{bmatrix} \bigg|_{\theta'=[0,0]} = \begin{bmatrix} 0 \\ 0 \end{bmatrix}
$$

##### **3. 计算 Hessian 矩阵 $H$**  
$$
H = \nabla^2 L(\theta') = \begin{bmatrix}
\frac{\partial^2 L}{\partial \theta_1^2} & \frac{\partial^2 L}{\partial \theta_1 \partial \theta_2} \\
\frac{\partial^2 L}{\partial \theta_2 \partial \theta_1} & \frac{\partial^2 L}{\partial \theta_2^2}
\end{bmatrix} = \begin{bmatrix} 2 & 1 \\ 1 & 4 \end{bmatrix}
$$
##### **4. 泰勒展开公式**  
在 $\theta' = [0, 0]^\top$ 处展开：
$$
L(\theta) \approx \underbrace{0}_{L(\theta')} + \underbrace{(\theta - 0)^\top \begin{bmatrix} 0 \\ 0 \end{bmatrix}}_{\text{一阶项}} + \frac{1}{2}(\theta - 0)^\top \begin{bmatrix} 2 & 1 \\ 1 & 4 \end{bmatrix} (\theta - 0)
$$


化简后：
$$
L(\theta) \approx \frac{1}{2}\theta^\top \begin{bmatrix} 2 & 1 \\ 1 & 4 \end{bmatrix} \theta = \frac{1}{2}(2\theta_1^2 + 2\theta_1\theta_2 + 4\theta_2^2)
$$


展开后与原函数一致：
$$
L(\theta) = \theta_1^2 + 2\theta_2^2 + \theta_1\theta_2
$$

