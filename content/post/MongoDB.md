---
title: MongoDB基本语法
params:
  author: 一只饺子
tags:
  - 大数据
  - 语法
categories: 大数据
image: post/img/9.png
---
### 增

```MongoDB
db.getCollection('uesr').insert({"userId" : "014","uclass" : "B","name" : "Back","age" : 11,"email" : "b14@sina.com","birthday" : ISODate("2018-07-31T03:46:13.885Z"),"dataStatus" : 1})
```

### 删

```MongoDB
db.getCollection('user').remove({"userId":"014"})
```

### 改

```MongoDB
db.getCollection('user').update({"userId":"013"},{$set:{"email":"b13@sina.com", "age":20}})
```

> 在MongoDB中，`$set`是一个更新操作符，用于修改文档中某个字段的值，或向文档中添加新的字段，而不会影响其他字段。


### 查

```MongoDB
db.getCollection('user').find({}); // 查询所有

db.getCollection('user').find({"uclass":"A"}); // 查询条件:=

db.getCollection('user').fing({"name":/Ba/}); // 查询条件:like

db.getCollection('user').distinct({"name"}); // 查询条件:distinct

db.getCollection('user').find({"age":"{$gt:16}"}) // 查询条件:$gt//greater than

db.getCollection('user').find({"uclass":{$in:['A', 'B']}}); // 查询条件: in

db.getCollection('user').find({"uclass":"B","age":{$gt:16}}) // 查询条件: and

db.getCollection('user').find({$or:[{"uclass":"A"},{"class":"B"}]});// 查询条件: or

db.getCollection('user').find({"birthday":{$gt: new Date("2008-08-14T06:24:40.110Z"), $lt: new Date("2015-08-14T06:14:40.089Z")}}); // 查询条件: 时间

db.getCollection('user').find({"uclass":"A"}).count(); // 查询条件: count

db.getCollection('user').find({}).sort({"age":1}); // 查询条件: sort升序

db.getCollection('user').find({}).sort({"age":-1}); // 查询条件: sort降序

db.getCollection('user').aggregate([{$group:{_id:"$uclass",num:{$sum:1}}}]); // 聚合查询: count单列

db.getCollection('user').aggregate([{$group:{_id:{uclass:"$uclass", age:"$age"},num:{$sum:1}}}]); // 聚合查询: count多列

db.getCollection('user').find({}).limit(5); // 分页查询: limit in

db.getCollection('user').find({}).limit(5).skip(5); // 分页查询: limit m, n

db.getCollection('user').find({}, {userId:1, name:1}); // 查询指定字段

db.getCollection('user').find({}, {dataStatus:0, _id:0}); // 排查指定字段
```

> 正则表达式语法：
> `/^Ba/`:匹配以`Ba`开头的字符串
> `/Ba$/`:匹配以`Ba`结尾的字符串
> `/[Bb]a/`:匹配`Ba`或`ba`
> `/ba/i`:查找`name`字段的值包含字符串`ba`的文档，不区分大小写

> distinct的意思是去重

> `$gt`: $>$
> `$gte`: $\ge$ 
> `$lt`: $\le$
> `$lte`: $\le$
> `$ne`: $!=$
> `$eq`: $==$
