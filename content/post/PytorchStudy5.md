---
title: "Pytorch实践（刘二大人）5：逻辑斯蒂回归"
tags:
- 机器学习
- 代码实践
categories: 机器学习
image: post/img/17.jpg
---



## 基础知识

### 交叉熵(Cross-Entropy)

交叉熵衡量的是估计的概率分布Q近似真实分布P时所需的平均信息量
$$
H(P,Q)=-\sum_i P(i)lnQ(i)
$$

### 似然函数(Likelihood Function)

表示给定模型参数$\theta$时，观察到当前数据集$D$的概率
$$
L(\theta;D)=P(D|\theta)
$$
**核心思想**：最大似然估计(MLE)：通过调整参数$\theta$，使当前数据出现的概率最大化



### 对数似然(Log-Likelihood)

连乘容易导致数值下溢或溢出，取对数将乘法转为加法
$$
lnL(\theta;D)=\sum_{i=1}^N lnP(y_i|x_i;\theta)
$$

### 损失函数(Loss Function)

在最大似然估计中，负对数似然常被用作损失函数
$$
\mathcal{L}(w,b)=-\sum_{i=1}^N[y_iln\hat{y_i}+(1-y_i)ln(1-\hat{y})]
$$


## 数学原理

### 模型结构：线性组合+Sigmoid函数

逻辑斯蒂回归的核心是将线性回归的输出映射到概率空间（0和1之间）

* **线性部分**：
  * 对于输入特征向量$\vec{x}=[x_1,x_2,...,x_n]$，计算线性组合：$z=\vec{w}^T\vec{x}+b=w_1x_1+w_2x_2+...+w_nx_n+b$
  * 其中，$\vec{w}$是权重向量，$b$是偏置项
* **Sigmoid函数**：
  * 将线性输出$z$通过Sigmoid函数转换为概率：$P(y=1|\vec{x})=\sigma(z)=\frac{1}{1+e^{-z}}$



### 损失函数：交叉熵

逻辑斯蒂回归通过**极大似然估计**（MLE）求解参数，对应的损失函数是**交叉熵损失**

* **似然函数**：对每个样本$(x_i,y_i)$，其似然为$P(y_i|x_i)=\sigma(z_i)^{y_i}\cdot (1-\sigma(z_i))^{1-y_i}$
* **对数似然与损失似然**：$\mathcal{L}(w,b)=-\sum_{i=1}^N [y_i ln\sigma(z_i)+(1-y_i)ln(1-\sigma(z_i))]$



### 参数优化：梯度下降

通过梯度下降法迭代更新权重$w$和偏置$b$

* **梯度计算**：Sigmoid函数的导数$\sigma '(z)=\sigma(z)(1-\sigma(z))$，损失函数$w$和$b$的梯度为：

$$
\frac{\partial \mathcal{L}}{\partial w_j}=\sum_{i=1}^{N}(\sigma(z_i)-y_i)x_{ij},\frac{\partial\mathcal{L}}{\partial b}=\sum_{i=1}{N}(\sigma(z_i)-y_i)
$$

* **参数更新**:

$$
w=w-\eta \frac{\partial \mathcal{L}}{\partial w}, b=b-\eta\frac{\partial \mathcal{L}}{\partial b}
$$

### 决策边界

逻辑斯蒂回归的决策边界是线性的，由方程$w^Tx+b=0$定义

* 当$\sigma(z)\ge 0.5$，预测$y=1$（即$z\ge 0$）；
* 否则预测$y=0$



## 思路

### 数据准备

```python
x_data = torch.tensor([[1.0],[2.0],[3.0]])
y_data = torch.tensor([[0],[0],[1]])
```

* **输入数据** `x_data`：3个样本，每个样本1个特征（形状为 `[3, 1]`）。
* **标签数据** `y_data`：对应的二分类标签（0或1）。
  * 当特征值为1.0和2.0时，标签是0；特征值为3.0时，标签是1。
  * 这可以理解为模型需要学习“当特征值大于某个阈值时预测为1”。



### 模型定义

```python
class LogisticRegressionModel(torch.nn.Module):
    def __init__(self):
        super(LogisticRegressionModel, self).__init()
        self.linear = torch.nn.Linear(1,1) # 表示输入和输出维度均为1
        
    def forward(self, x):
        y_pred = torch.sigmoid(self.Linear(x))
        return y_pred
```



### 损失函数与优化器

```python
criterion = torch.nn.BCELoss(size_average=False)  # 二元交叉熵损失（累加模式）
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # 随机梯度下降
```

* **损失函数** `BCELoss`：二元交叉熵损失（Binary Cross Entropy Loss），用于衡量预测概率与真实标签的差异。
  * `size_average=False` 表示损失是**累加**而非平均（PyTorch新版本中已更名为 `reduction='sum'`）。
* **优化器** `SGD`：随机梯度下降，学习率 `lr=0.01`。



### 训练循环

```python
for epoch in range(1000):
    y_pred = model(x_data)          # 前向传播
    loss = criterion(y_pred, y_data)  # 计算损失
    optimizer.zero_grad()           # 清空梯度
    loss.backward()                 # 反向传播计算梯度
    optimizer.step()                # 更新参数（w和b）
```



## 代码实现

```python
import torch

x_data = torch.tensor([[1.0], [2.0], [3.0]])
y_data = torch.tensor([[2.0], [4.0], [6.0]])

class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

model = LinearModel()

criterion = torch.nn.MSELoss(reduction = 'sum')
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)

for epoch in range(100):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(epoch, loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print('w = ', model.linear.weight.item())
print('b = ', model.linear.bias.item())

x_test = torch.tensor([[4.0]])
y_test = model(x_test)
print('y_pred = ', y_test.data)
```

