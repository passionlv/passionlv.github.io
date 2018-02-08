---
layout: post
title:  "基于sklearn和python的机器学习-视频教程"
categories: 学习
tags:  学习笔记 机器学习 视频教程
author: xiaopeng
---

* content
{:toc}

Data School的Kevin Markham讲解的10次课程。语速慢，配合youtube自动字幕，使得英文听课全无压力。此外，Kevin还有30讲的pandas教程，值得推荐，在B站和Youtube都有。




##  基于sklearn和python的机器学习-视频教程

### 0. 教程概况

讲解人介绍：Data School的Kevin Markham讲解的10次课程。语速慢，配合youtube自动字幕，使得英文听课全无压力。此外，Kevin还有30讲的pandas教程，值得推荐，在B站和Youtube都有。

课程时间：2015-2016年

课程资源：
- 课程视频地址： [Youtube地址](http://bit.ly/scikit-learn-videos)
- 课程源代码 [Github地址](https://github.com/justmarkham/scikit-learn-videos)
- 课程 [在线jupyter](https://nbviewer.jupyter.org/github/justmarkham/scikit-learn-videos/tree/master/)， 使用vbviewer网站提供的在线jupyter，非常不错的网站。 此外本课程在可以通过 [binder](https://hub.mybinder.org/user/justmarkham-scikit-learn-videos-whvo30at/tree)  网站在线加载notebook，无需下载情况下在线运行。
- 课程 [Blog](http://blog.kaggle.com/2015/04/08/new-video-series-introduction-to-machine-learning-with-scikit-learn/)  blog里面视频播放地址、课程主要内容讲义、其他资料链接，非常有用。 当然作者github上的Notebook文件内容更详细和完整。

##### 主要内容（主要翻译自作者github README）
1. 机器学习概念及工作原理
2. 用python的机器学习工具
3. iris数据集
4. 如何训练模型
5. 模型训练中的参数调试
6. 建模三部曲，pandas读取数据，seaborn可视化数据，scikit-learn建模
7.  交叉验证、参数调试、模型选择和特征选择
8. 高效搜索最优参数
9. 分类模型的评估
10. 基于文本数据的分析

##### 教程知识要点

##### 学习总结

##### 个人感受



### 1. 什么是机器学习，它如何工作？ ([视频](https://www.youtube.com/watch?v=elojMnjn4kk&list=PL5-da3qGB5ICeMbQuqbbCOQWcS6OYBr5A&index=1), [代码](01_machine_learning_intro.ipynb), [博客](http://blog.kaggle.com/2015/04/08/new-video-series-introduction-to-machine-learning-with-scikit-learn/))
#### 主要内容
  - 什么是机器学习?
  - 机器学习的两个主要类别是什么?
  - 机器学习实例
  - 机器学习如何工作?

### 2. 使用Python进行机器学习: scikit-learn and IPython Notebook ([视频](https://www.youtube.com/watch?v=IsXXlYVBt1M&list=PL5-da3qGB5ICeMbQuqbbCOQWcS6OYBr5A&index=2), [代码](02_machine_learning_setup.ipynb), [博客](http://blog.kaggle.com/2015/04/15/scikit-learn-video-2-setting-up-python-for-machine-learning/))
#### 主要内容
  - scikit-learn优缺点
  - scikit-learn安装
  - IPython Notebook使用
  - Python的学习资源

### 3. 通过著名的iris数据集开始scikit-learn ([video](https://www.youtube.com/watch?v=hd1W4CyPX58&list=PL5-da3qGB5ICeMbQuqbbCOQWcS6OYBr5A&index=3), [notebook](03_getting_started_with_iris.ipynb), [blog post](http://blog.kaggle.com/2015/04/22/scikit-learn-video-3-machine-learning-first-steps-with-the-iris-dataset/))
#### 主要内容
  - 著名的iris数据集，以及其与机器学习的关系
  - 在scikit-learn加载数据集?
  - 如何用机器学习术语描述数据集
  - scikit-learn对分析数据的四个关键的需求

### 4. 通过scikit-learn训练模型 ([video](https://www.youtube.com/watch?v=RlQuVL6-qe8&list=PL5-da3qGB5ICeMbQuqbbCOQWcS6OYBr5A&index=4), [notebook](04_model_training.ipynb), [blog post](http://blog.kaggle.com/2015/04/30/scikit-learn-video-4-model-training-and-prediction-with-k-nearest-neighbors/))
#### 主要内容
  - K-nearest neighbors分类模型
    - 设定一个K值
    - 在训练集中，选择与未知数据最近的k个数据
    - 在k个中，选择个数最多的类别，作为未知数据的预测类别
    - 随着k的变化，类别之间的分界线或决策线，会发生变化。
  - scikit-learn中训练模型的四个步骤
    - import要用的模型类（model class）
    - **实例化** 一个模型对象，作为评估者（estimator，为sklearn对模型的称呼，表示评估未知的实物）。实例化需要注意:
      - estimator对象的名字无要求，但通常用model名称以便于增加可读性
      - 参数非常关键，通常是模型的超参数或可调参数
      - 未明确赋值的参数，将采用默认值。sklearn提供了合理的默认参数，在即使不调参的情况下，仍有不错表现。打印模型，print(knn),将输出模型的所有参数。      
    -  用data去fit模型，也是模型训练步骤，该步骤模型去学习特征与结果标签之间的关系。fit步骤是更新（inplace）模型，因此无需在赋值给另外一个对象。
    - 用训练好的模型去predict未知数据。未知数据可以是一条或多条数据。
  - 模型调试：
    - 重复上述步骤2-4
  - 利用四个步骤在其他模型中训练：
    - 上述步骤的接口对于所有的模型都是一致的。
课后kevin给出了另外一个关于算法的视频课程，值得学习。
### 5. 比较scikit-learn中的训练模型 ([video](https://www.youtube.com/watch?v=0pP4EwWJgIU&list=PL5-da3qGB5ICeMbQuqbbCOQWcS6OYBr5A&index=5), [notebook](05_model_evaluation.ipynb), [blog post](http://blog.kaggle.com/2015/05/14/scikit-learn-video-5-choosing-a-machine-learning-model/))
#### 主要内容
  - 如何为监督学习任务选择模型，两个办法：
    - 在整个数据集上训练和测试模型

  - 如何为模型调试最优的参数
  - 如何在样本之外，评估模型近似性能

### 6. 数据科学处理流程: pandas, seaborn, scikit-learn ([video](https://www.youtube.com/watch?v=3ZWuPVWq7p4&list=PL5-da3qGB5ICeMbQuqbbCOQWcS6OYBr5A&index=6), [notebook](06_linear_regression.ipynb), [blog post](http://blog.kaggle.com/2015/05/28/scikit-learn-video-6-linear-regression-plus-pandas-seaborn/))
#### 主要内容
  - 如何用pandas读取数据
  - 如何用seaborn对数据进行可视化
  - 线性回归以及其如何工作
  - scikit-learn中训练和理解线性回归模型
  - 回归问题的评估指标
  - 模型中的特征选择

### 7. 交叉验证、参数调试、模型选择和特征选择 ([video](https://www.youtube.com/watch?v=6dbrR-WymjI&list=PL5-da3qGB5ICeMbQuqbbCOQWcS6OYBr5A&index=7), [notebook](07_cross_validation.ipynb), [blog post](http://blog.kaggle.com/2015/06/29/scikit-learn-video-7-optimizing-your-model-with-cross-validation/))
#### 主要内容
  - 评估模型中，训练和测试分离的缺点
  - 如何克服k-折交叉验证的限制
  - 用交叉验证来调试参数，选择模型和选择特征
  - 交叉验证的一些提升

### 8. 高效搜索最优参数 ([video](https://www.youtube.com/watch?v=Gol_qOgRqfA&list=PL5-da3qGB5ICeMbQuqbbCOQWcS6OYBr5A&index=8), [notebook](08_grid_search.ipynb), [blog post](http://blog.kaggle.com/2015/07/16/scikit-learn-video-8-efficiently-searching-for-optimal-tuning-parameters/))
#### 主要内容
  - 用k-折交叉验证搜索最优参数
  - 提升搜索性能
  - 一次搜索多个最优参数
  - 在预测前，如何处理最优参数W
  - 减少搜索参数的计算开销

### 9. 分类模型的评估 ([video](https://www.youtube.com/watch?v=85dtiMz9tSo&list=PL5-da3qGB5ICeMbQuqbbCOQWcS6OYBr5A&index=9), [notebook](09_classification_metrics.ipynb), [blog post](http://blog.kaggle.com/2015/10/23/scikit-learn-video-9-better-evaluation-of-classification-models/))
#### 主要内容
  - 模型评估的目的，以及常用评估过程
  - 分类准确性使用和限制
  - 混淆矩阵（confusion matrix）描述分类器性能
  - 混淆矩阵（confusion matrix）衍生的其他矩阵
  - 通过改变分类的阈值，调整分离器性能
  - ROC曲线作用
  - Area Under the Curve (AUC)与分类精确性的区别

### 基于文本的数据分析

PyCon 2016会议上，老师讲解的3个小时的基于文本数据的分析。 [tutorial video](https://www.youtube.com/watch?v=ZiKMIuYidY0&list=PL5-da3qGB5ICeMbQuqbbCOQWcS6OYBr5A&index=10)
#### 主要内容
1. scikit-learn建模 (复习)
2. 用数字数据表示文本数据
3. 用pandas提取分本数据
4. 数据矢量化
5. 模型创建和评估
6. 模型比较
7. 检查模型从而深入理解模型Examining a model for further insight
8. 在其他数据集中实践建模工作流程
9. 调参和矢量化(讨论)
