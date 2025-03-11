---
title: "Pytorch实践（刘二大人）2：梯度下降算法"
tags:
- 机器学习
- 代码实践
categories: 机器学习
image: post/img/14.jpg
---

## 思路

1. **模型定义**，如假设模型为$y=w \cdot x$
2. **损失函数**：$MSE = \frac{1}{N}\sum_{i=1}^{N}(w\cdot x_i - y_i)^2$

3. **梯度计算**：$\frac{\partial \text{MSE}}{\partial w} = \frac{2}{N} \sum_{i=1}^{N} x_i \cdot (w \cdot x_i - y_i)$
   1. **推导**：$\frac{\partial (wx_i-y_i)^2}{\partial w} = 2x_i(wx_i-y_i)$



4. **更新$w$**： $w_{new} = w_{old}-\alpha \cdot \frac{\partial MSE}{\partial w}$

5. **开始训练**



## 数学推导

1. **单变量**，如$f(x)=\frac{1}{2}x^2\Longrightarrow \nabla x=\frac{\partial f(x)}{\partial x}=x$
2. **多变量**，如$f_1(\theta)=2\theta_1^2+3\theta_2^2+4\theta_3^2\Longrightarrow \nabla \theta = (4\theta_1,6\theta_2,8\theta_3)$
3. **迭代公式**：$x^{k+1}=x^k - \lambda\nabla f(x^k)$[^1]
   1. 由**一阶泰勒展开**，假设当前参数为$x^k$，在附近寻找一个更小的函数值$f(x^{k+1})$，可近似为$f(x^{k+1})\approx f(x^k)+\nabla f(x^k)^T(x^{k+1}-x^k)$
   2. 为了最小化$f(x^{k+1})$，需要选择$x^{k+1}$使得：$x^{k+1}-x^k=-\lambda\nabla f(x^k)$



## 代码实现

```python
import matplotlib.pyplot as plt

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

# 初始化假设w的值
w = 1.0


# 线性模型为y = w * x
def forward(x):
    return x * w


# 损失函数
def cost(xs, ys):
    cost = 0
    for x, y in zip(xs, ys):
        y_pred = forward(x)
        cost += (y_pred - y) ** 2
    return cost / len(xs)


# 梯度下降函数
def gradient(xs, ys):
    grad = 0
    for x, y in zip(xs, ys):
        grad += 2 * x * (x * w - y)
    return grad / len(xs)


epoch_list = []
cost_list = []
print('predict (before training)', 4, forward(4))
for epoch in range(100):
    cost_val = cost(x_data, y_data)
    grad_val = gradient(x_data, y_data)
    w -= 0.01 * grad_val  # 0.01 learning rate
    print('epoch:', epoch, 'w=', w, 'loss=', cost_val)
    epoch_list.append(epoch)
    cost_list.append(cost_val)

print('predict (after training)', 4, forward(4))
plt.plot(epoch_list, cost_list)
plt.ylabel('cost')
plt.xlabel('epoch')
plt.show()

```

> 在梯度下降法中，参数最终只能**逼近理论最优值**而不会完全等于它.因为**导数为零的点是理论解**，实际计算中梯度只能趋近于零，但不会严格为零



> 损失函数(Loss Function)和代价函数(Cost Function)经常交叉使用，但是损失函数是用于衡量模型对**单个样本**的预测值与真实值之间的差异，代价函数是用于衡量模型对**整个训练集**的预测值与真实值之间的总体误差。

[^1]: 这里的$x^k$表示第$k$次迭代，与幂运算无关
