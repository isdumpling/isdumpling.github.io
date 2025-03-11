---
title: "PyTorch语法"
tags:
- 机器学习
- 语法
- 代码实践
categories: 机器学习
typora-root-url: ./..\..\static
image: post/img/12.jpg
---

## 在Anaconda中使用PyTorch

1. 查看CUDA版本的三种方法

```
nvcc -V
nvcc --version
nvidia -smi
```

2. 快速搭建虚拟环境

```bash
# 创建环境并指定Python版本
conda create -n <env_name> python=3.9 

# 激活环境
conda activate <env_name>

# 安装PyTorch版本
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# 退出虚拟环境
conda deactivate
```

3. 使用Python代码进行验证是否安装成功torch

```python
import torch # 如果pytorch安装成功即可导入
print(torch.cuda.is_available) # 查看CUDA是否可用
print(torch.cuda.device_count) # 查看可用的CUDA数量
print(torch.version.cuda) # 查看CUDA的版本号
```



## 张量（Tensor）

在PyTorch中，张量类似于Numpy中的数组，但PyTorch中的张量可以运行在不同设备，如CPU和GPU，Numpy数组只能在CPU上运行

* **维度（Dimensionality）**：张量的维度指的是数据的多维数组结构。例如，一个标量（0维张量）是一个单独的数字，一个向量（1维张量）是一个一维数组，一个矩阵（2维张量）是一个二维数组，以此类推。
* **形状（Shape）**：张量的形状是指每个维度上的大小。例如，一个形状为`(3, 4)`的张量意味着它有3行4列。
* **数据类型（Dtype）**：张量中的数据类型定义了存储每个元素所需的内存大小和解释方式。PyTorch支持多种数据类型，包括整数型（如`torch.int8`、`torch.int32`）、浮点型（如`torch.float32`、`torch.float64`）和布尔型（`torch.bool`）。



### 张量创建

```python
import torch

# 创建一个 2x3 的全 0 张量
a = torch.zeros(2, 3)
print(a)

# 创建一个 2x3 的全 1 张量
b = torch.ones(2, 3)
print(b)

# 创建一个 2x3 的随机数张量
c = torch.randn(2, 3)
print(c)

# 从 NumPy 数组创建张量
import numpy as np
numpy_array = np.array([[1, 2], [3, 4]])
tensor_from_numpy = torch.from_numpy(numpy_array)
print(tensor_from_numpy)

# 在指定设备（CPU/GPU）上创建张量
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
d = torch.randn(2, 3, device=device)
print(d)
```



输出结果

![输出结果](/img/tensor_output.jpg)



### 常见张量操作

```python
# 张量相加
e = torch.randn(2, 3)
f = torch.randn(2, 3)
print(e + f)

# 逐元素乘法
print(e * f)

# 张量的转置
g = torch.randn(3, 2)
print(g.t())

# 张量的形状
print(g.shape)
```

 

输出结果

![输出结果](/img/tensor_output1.jpg)

`item()`方法

```python
# 将包含单个元素的张量转换为Python标量
print(forward(4)) # 输出：tensor([0.8], grad_fn=<SigmoidBackward>)
print(forward(4).item()) # 输出：0.8
```

> 仅限单元素张量。如果张量包含多个元素，调用`item()`会报错



## 自动求导(Autograd)

自动求导允许计算机自动计算数学函数的导数

在深度学习中，自动求导主要用于两个方面

1. 训练神经网络时计算梯度
2. 进行反向传播算法的实现

**动态图与静态图**

* **动态图(Dynamic Graph)**：在动态图中，计算图在运行时动态构建。每次执行操作时，计算图都会更新，这使得调试和修改模型变得更加容易。PyTorch使用的是动态图。
* **静态图（Static Graph）**：在静态图中，计算图在开始执行之前构建完成，并且不会改变。TensorFlow最初使用的是静态图，但后来也支持动态图。



```python
# 创建一个需要计算梯度的张量
x = torch.randn(2, 2, requires_grad=True)
print(x)

# 执行某些操作
y = x + 2
z = y * y * 3
out = z.mean()

print(out)
```

输出结果

![输出结果](/img/autograd.jpg)



### 反向传播(Backpropagation)

一旦定义了计算图，可以通过`.backward()`方法来计算梯度

```pytho
# 反向传播，计算梯度
out.backward()

# 查看x的梯度
print(x.gred)
```

输出结果

![输出结果](/img/autograd1.jpg)



### 停止梯度计算

如果你不希望某些张量的梯度被计算（例如，当你不需要反向传播时），可以使用 `torch.no_grad()` 或设置 `requires_grad=False`。

```python
# 使用 torch.no_grad() 禁用梯度计算
with torch.no_grad():
    y = x * 2
```

