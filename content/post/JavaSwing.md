---
title: "Java Swing"
params:
  author: 一只饺子
date: 2025-02-25T17:21:44+08:00
tags:
  - Java
  - 语法
categories: Java
draft: true
---

## 简介

Java Swing 是 Java 用于构建 **图形用户界面（GUI）** 的核心工具包，它是 Java Foundation Classes（JFC）的一部分。

**Swing与AWT的关系**

| 特性         | AWT                     | Swing              |
| ------------ | ----------------------- | ------------------ |
| 实现方式     | 依赖本地GUI库           | 纯Java绘制         |
| 组件类型     | 重量级                  | 轻量级             |
| 跨平台一致性 | 低（外观随系统变化）    | 高（可统一风格）   |
| 性能         | 较高（直接调用系统API） | 较低（纯Java绘制） |
| 组件丰富度   | 基础组件                | 高级组件           |

> 现代开发中通常使用 Swing 而非 AWT，但 AWT 的布局管理器（如 `BorderLayout`）仍常与 Swing 配合使用。



**Swing组件层级结构**

```Java
java.lang.Object
   └─ java.awt.Component
       └─ java.awt.Container
           └─ javax.swing.JComponent
               ├─ JButton
               ├─ JLabel
               └─ ...

```



## Swing实战

```Java
import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JPasswordField;
```

