---
title: xgboost 
tags: 集成算法
grammar_cjkRuby: true
---


### 与决策树结合：

![决策树][1]

		一个决策树效果不是很好，使用两个决策树集成提高效果
		这样，讲一个分类器称为弱分类器，将结合后的分类器称为抢分类器

![结合][2]

				λ可自行指定

#### 基本原理：

![xgboost][3]

![思路][4]

		希望每添加一个分类器（决策树）效果会有相应的提升
		xgboost是一种集成算法，提升算法
		核心：保留原有分类器效果的同时不断添加新的分类器提高效果

![新添加模型的选取][5]

		l表示真实值与预测值之间的误差,Ω的累加是正则化的惩罚项
![转化][6]

![求解][7]

![求解变换][8]

![整体表示][9]

![求偏导求解][10]

		G,H的计算需要给定loss函数

![举例][11]

![举例的解题思路][12]


  [1]: ./images/1514276441315.jpg
  [2]: ./images/1514277072085.jpg
  [3]: ./images/1514276628601.jpg
  [4]: ./images/1514276771728.jpg
  [5]: ./images/1514277175324.jpg
  [6]: ./images/1514277420456.jpg
  [7]: ./images/1514277581338.jpg
  [8]: ./images/1514277832191.jpg
  [9]: ./images/1514278610065.jpg
  [10]: ./images/1514278662256.jpg
  [11]: ./images/1514278825525.jpg
  [12]: ./images/1514278954010.jpg