---
title: "Pytorch实践（刘二大人）1：线性模型"
tags:
- 机器学习
- 代码实践
categories: 机器学习
image: post/img/13.jpg
---

## 使用穷举法寻找线性回归模型中最佳权重参数`w`

注意：

* 对于`numpy`需要先下载依赖`pip install numpy`
* 对于`matplotlib.pyplot`，若有`anaconda`则无需下载

```python
import numpy as np;
import matplotlib.pyplot as plt

x_data = [1.0,2.0,3.0]
y_data = [2.0,4.0,6.0]

w_list = []
mse_list = []

# 定义前馈函数
def forward(x):
    return x * w

# 定义损失函数
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2

for w in np.arange(0.0, 4.0, 0.1):
    print("w = ", w)
    loss_sum = 0
    # zip函数：将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表
    for x_val, y_val in zip(x_data, y_data):
        y_pred_val = forward(x_val)
        loss_val = loss(x_val, y_val)
        loss_sum += loss_val
        print('\t',x_val,y_val,y_pred_val,loss_val)
    print('MSE = ', loss_sum / 3)
    w_list.append(w)
    mse_list.append(loss_sum / 3)

plt.plot(w_list, mse_list)
plt.xlabel("w")
plt.ylabel("loss")
plt.show()

```



## 实现线性模型并输出loss的3D图形

### 思路

1. 设计线性模型$y=\omega x+b$
2. 预估$\omega,b$的范围并用$W,B$数组存储
3. $\omega=1_n \cdot W,b=B\cdot 1_m^T$构成参数空间$(w,b)$的笛卡尔积网络
4. $L_{sum}=0_{n\times m}$
5. $MSE=\frac{1}{N}\sum$



```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

x_data = np.array([1.0, 2.0, 3.0])
y_data = np.array([5.0, 8.0, 11.0])

def forward(x):
    return x * w + b

def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2

W = np.arange(0.0, 4.1, 0.1)
B = np.arange(0.0, 4.1, 0.1)
w, b = np.meshgrid(W, B)

loss_sum = np.zeros_like(w)

for x_val, y_val in zip(x_data, y_data):
    loss_val = loss(x_val,y_val)
    loss_sum += loss_val

mse = loss_sum / len(x_data)

fig = plt.figure() # 创建一个新的窗口
ax = fig.add_subplot(111,projection='3d') # 
ax.plot_surface(w,b,mse,cmap='viridis') # camp='viridis'指定颜色映射
ax.set_xlabel('w')
ax.set_ylabel('b')
ax.set_zlabel('MSE')
plt.show()
```

