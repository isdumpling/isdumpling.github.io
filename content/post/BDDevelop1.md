---
title: '大数据开发技术第一章'
tags:
- 大数据
- 大数据开发技术
categories: 大数据
image: post/img/1.jpg
---

> 说明：本笔记内容基于NIIT PPT，加之个人理解

## 1A

### Big Data Concept

Big Data is a collection of data that is huge in volume, yet growing exponentially with time



### Types of Big Data

* Structured
* Unstructured
* Semi-structured



### Characteristics of Big Data

* Volume
* Velocity
* Variety
* Veracity



### Advantages of Big Data Processing

* Businesses
* Improved customer service
* Early identification of risk
* Better operational efficiency



### Key Vocabulary

> 说明：由于个人认为NIIT一些中文翻译不妥，随自行翻译一遍

| English           | Chinese(NIIT) | Chinese(Dumpling)          |
| ----------------- | ------------- | -------------------------- |
| Exponential       | 指数型        | 指数级（增长）             |
| Generates         | 产生          | 生成                       |
| Processed         | 处理          | 处理                       |
| Structured        | 结构化的      | 结构化                     |
| Unstructured      | 处理非结构化  | 非结构化                   |
| Semi-structured   | 处理半结构化  | 半结构化                   |
| Enormous          | 巨大的        | 海量的                     |
| Heterogeneous[^1] | 异质          | 异构的                     |
| Analyzing         | 分析          | 分析                       |
| Volume            | 体积          | 数据体量                   |
| Velocity          | 种类          | 处理速度                   |
| Variety           | 速度          | 数据多样性                 |
| Variety           | 真实性        | 数据多样性（重复项需校对） |
| Intelligence      | 智力          | 智能                       |

[^1]: 异构，指系统中存在多种不同形式的组成部分

## 1B

### Hadoop Ecosystem tools

* **Data Storage**
  * HDFS(File System)
  * HBASE(File System)
* **Data Processing**
  * Map Reduce(Cluster Management)
  * YARN(Cluster & Resource Management)
* **Data Access**
  * Hive(SQL)
  * Pig(Dataflow)
  * Mahout(Machine Learning)
  * Avro(RPC)
  * Sqoop(RDBMS Connector)
* **Data Management**
  * Oozie(Workfolw Monitering)
  * Chukwa(Monitoring)
  * Flume(Monitoring)
  * ZooKeeper(Management)



**Hadoop生态协同流程示例**：

1. **存储**：用HDFS存原始数据，HBase存需快速访问的数据
2. **处理**：YARN调度资源，MapReduce做离线计算
3. **访问**：Hive执行SQL查询，Sqoop导出结果到MySQL
4. **管理**：Oozie调度任务链，ZooKeeper确保服务高可用



#### HBASE

* HBase is an open-source, non-relational distributed database.
* Modelled agter Google's BigTable
* The HBase was designed to run on top of HDFS and procides BigTable-like capabilities
* Written in Java



#### HIVE

* The Apache Hive data warehouse software is built on Apache Hadoop to query and manage large distributed data sets
* Hive defines a query language similar to SQL, called HQL



#### Hadoop

* HDFS is a specially designed file system for storing huge datasets in commodity hardware,storing information in different formats on various machines



#### Apache Storm

* Apache Storm is a free, open source distributed real-time computing system that simplifies the reliable processing of streaming data
* Easy to expand
* Storm fault tolerance
* Low latency



#### ZooKeeper

* Apache Zookeeper is the coordinator of any Hadoop job which includes a combination of various services in a Hadoop Ecosystem
* Apache Zookeeper coordinates with various services in a distributed environment



#### Sqoop

* Allows users to extract data from relational databases into Hadoop for further processing
* The import process runs a MapReduce job that connects to the MySQL database and reads data from tables



### Criteria to Evaluate Distribution

* Performance
* Scalability
* Reliability



### Key Vocabulary

| English         | Chinese(NIIT) | Chinese(Dumpling) |
| --------------- | ------------- | ----------------- |
| Ecosystem       | 生态系统      | 生态系统          |
| Fault-tolerant  | 容错          | 容错性            |
| Latency         | 潜伏          | 延迟              |
| Configuration   | 配置          | 配置              |
| Synchronization | 同步化        | 同步              |
| Scalability     | 可拓展性      | 可扩展性          |
| Bottlenecks     | 瓶颈          | 瓶颈              |
| Clustered       | 成簇的        | 集群化            |
| Replication     | 复制          | 数据复制          |
| Analytics       | 分析          | 数据分析          |

