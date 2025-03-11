---
title: "Pytorch实践（刘二大人）4：用PyTorch实现线性回归"
tags: 
- 机器学习
- 代码实践
categories: 机器学习
image: post/img/16.jpg
---



## 思路

### 准备数据

```python
x_data = torch.tensor([[1.0], [2.0], [3.0]])
y_data = torch.tensor([[2.0], [4.0], [6.0]])
```

### 定义模型结构

* 定义线性模型$y = w \times x + b$

```python
class LinearModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)  # 输入维度1，输出维度1（即 y = w*x + b）

    def forward(self, x):
        y_pred = self.linear(x)  # 计算预测值
        return y_pred

model = LinearModel()  # 创建模型实例
```

* `torch.nn.Linear(1, 1)`：PyTorch提供的线性层，自动初始化`w`和`b`（比如 `w=0.5`, `b=0.3`，随机值）

### 定义损失函数和优化器

```python
criterion = torch.nn.MSELoss(reduction='sum')  # 均方误差损失（所有样本的误差平方和）
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # 随机梯度下降优化器
```

* **损失函数**：例如，如果模型预测 `y_pred = [1.5, 3.0, 4.5]`，正确值是 `[2,4,6]`，则损失为 `(1.5-2)^2 + (3-4)^2 + (4.5-6)^2`
* **优化器**：`lr=0.01` 是学习率（每次调整的步长）。



### 训练循环

#### 前向传播：

```python
y_pred = model(x_data)
```

用当前的`w`和`b`计算预测值



#### 计算损失

```python
loss = criterion(y_pred, y_data)
```



#### 反向传播

```python
optimize.zero_grad()
loss.backward()
```



#### 更新参数

```python
optimizer.step()
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

