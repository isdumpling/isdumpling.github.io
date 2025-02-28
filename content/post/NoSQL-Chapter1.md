---
title: NoSQL Chapter 1
params:
  author: 一只饺子
date: 2025-02-25T16:32:11+08:00
tags:
- 大数据
categories: 大数据
---

### What is NoSQL?



* **NoSQL** stands for **Not Only SQL**
* Designed for **unstructured** (such as `video`, `audio`) and **semi-structured** (such as `json`, `csv`)
* **Flexible data models**, **horizontal scaling**, and **high performance**
  * **Flexible data models**:
  	* Key-value
  	* Graph
  	* Document
  	* Column Store
  * **Horizontal scaling**: data scalings are stored in mang machines, and throw adds nodes to expand the ability of computing
  * **High Performance**: For example, Redis stores data in memory, and its read and write speed can reach the microsecond level.



### Types of NoSQL Databases



* Key-value database
* Document database
* Wide-column database
* Graph database



### NoSQL vs RDBMS



* **NoSQL**: Flexible schema, horizontal scaling, high availability
* **RDBMS**: Rigid schema, vertical scaling, ACID compilance
	* ACID:  Atomicity, Consistency, Isolation, Durability



### Key Features of NoSQL

* Schema flexibility
* Horizontal scalability
* High availability and fault tolerance
* Diverse data models
* Cost-effectiveness



### MongoDB Overview

* Document-oriented NoSQL database
* Stores data in JSON-like BSON(Binary Json) format
* Flexible schema, horizontal scaling, and high performance



### MongoDB Features



* Document-oriented storage
* Flexible schema
* Scalability(sharding)
* Indexing and aggregation framework
* Replication and high availability



### MongoDB vs RDBMS



* **MongoDB**: Document-based, dynamic schema, horizontal scaling
* **RDBMS**: Row-based, rigid schema, vertical scaling



### Redis Overview



* In-memory data structure store
* Used as a database, cache, and message broker
* Supports strings, hashes, lists, sets, and stored sets



### Redis Features



* **Speed**: In-memory storage for fast readd/write operations
* **Persistence**: RDB and AOF for data durability
* **Replication**: Master-slave replication for high avaliablity
* **Luad scripting**: Complex operations can be executed atomically

> **RDB**: Redis Database. Snapshot is the process by which Redis saves the data in <mark>memory</mark> to the <mark>disk</mark> in the form of a binary file at a <mark>certain point</mark> in time. This binary file is the snapshot file.
> **Master-slave Replication**: It allows the data of one redis instance (<mark>master node</mark>) to be automatically to one or more other Redis instances (<mark>slave nodes</mark>).



### Redis vs MongoDB



* **Redis**: Key-value store, in-memory, fast read/write
* MongoDB: Document store, on-disk, flexible schema



### In-memory Databases



* Data stored in RAM for low latency and high throughput
* Ideal for real-time applications like gaming and analytics
* Redis is a popular in-memory database



### Redis History



* Created in 2009 by Salvatore Sanfilippo
* Key releases: Redis 3.0 (Cluster mode), Redis 6.0 (ACL, multi-threaded I/O).
* Used by companies like Twitter, Airbnb, and GitHub



### When to Use NoSQL



* Large volumes of unstructured data
* Need for horizontal scaling
* Real-time applications with low latency
* Dynamic data models

