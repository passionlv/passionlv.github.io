---
layout: post
title:  "基于sklearn和python的机器学习-视频教程"
categories: 学习
tags:  学习笔记 机器学习 视频教程 sklearn python
author: xiaopeng
---

* content
{:toc}

Data School的Kevin Markham讲解的10次课程。语速慢，配合youtube自动字幕，使得英文听课全无压力。此外，Kevin还有30讲的pandas教程，值得推荐，在B站和Youtube都有。




##  基于sklearn和python的机器学习-视频教程

### 0. 教程概况

讲解人介绍： [Data School](http://www.dataschool.io/)的Kevin Markham讲解的10次课程。语速慢，配合youtube自动字幕，使得英文听课全无压力。此外，Kevin还有30讲的pandas教程，值得推荐，在B站和Youtube都有。

课程时间：2015-2016年

##### 课程资源：
- [Data School官方网站](http://www.dataschool.io/)
- 课程视频地址： [Youtube地址](http://bit.ly/scikit-learn-videos)
- 课程源代码 [Github地址](https://github.com/justmarkham/scikit-learn-videos)
- 课程 [在线jupyter](https://nbviewer.jupyter.org/github/justmarkham/scikit-learn-videos/tree/master/)， 使用vbviewer网站提供的在线jupyter，非常不错的网站。 此外本课程在可以通过 [binder](https://hub.mybinder.org/user/justmarkham-scikit-learn-videos-whvo30at/tree)  网站在线加载notebook，无需下载情况下在线运行。
- 课程 [Blog](http://blog.kaggle.com/2015/04/08/new-video-series-introduction-to-machine-learning-with-scikit-learn/)  blog里面视频播放地址、课程主要内容讲义、其他资料链接，非常有用。 当然作者github上的Notebook文件内容更详细和完整。

##### 课外资(Kevin推荐的资料)：

- An Introduction to Statistical Learning (book): http://www-bcf.usc.edu/~gareth/ISL/
- Learning Paradigms (video): http://work.caltech.edu/library/014.html



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
  - 机器学习的两个主要类别：监督学习和非监督学习，本视频主要为监督学习。
  - 机器学习实例
  - 机器学习如何工作?

### 2. 使用Python进行机器学习: scikit-learn and IPython Notebook ([视频](https://www.youtube.com/watch?v=IsXXlYVBt1M&list=PL5-da3qGB5ICeMbQuqbbCOQWcS6OYBr5A&index=2), [代码](02_machine_learning_setup.ipynb), [博客](http://blog.kaggle.com/2015/04/15/scikit-learn-video-2-setting-up-python-for-machine-learning/))
#### 主要内容
  - scikit-learn优缺点
    - 优点1：机器学习的模型接口都是统一和一致的。例如fix，predict等
    - 优点2：提供很多参数以便于调参，同时还设定了常见的默认值
    - 优点3：非常好的文档
    - 优点4：有丰富的函数集，用于参数优化、数据处理、。。。
    - 优点5：活跃的社区，用户群，在stack overflow中很多人在提问和回答
    - 缺点1： 对于初学者有难度
    - 缺点2： 相比较R语言来说，强调模型调参，而不是模型的理解。偏实用，而不是偏解释说明。
  - scikit-learn安装：建议直接用Anaconda，我一直也是这么用的。
  - IPython Notebook使用：说明了几个快捷键，并建议参考官方指南
  - Python的学习资源
    - 讲义中介绍了几个课程，这里不再赘述。

### 3. 通过著名的iris数据集开始scikit-learn ([video](https://www.youtube.com/watch?v=hd1W4CyPX58&list=PL5-da3qGB5ICeMbQuqbbCOQWcS6OYBr5A&index=3), [notebook](03_getting_started_with_iris.ipynb), [blog post](http://blog.kaggle.com/2015/04/22/scikit-learn-video-3-machine-learning-first-steps-with-the-iris-dataset/))
#### 主要内容
  - 著名的iris数据集，以及其与机器学习的关系
  - 在scikit-learn加载数据集?
    - 可以在UCI网站在线下载或者用sklearn.datasets中load_iris    
  - 如何用机器学习术语描述数据集
    - 行是observation，也被称为样本，例子，实例或记录
    - 列是feature，也称为特征，属性，独立变量，输入等。(also known as: predictor, attribute, independent variable, input, regressor, covariate)
    - 结果标签，也被称为标签，目标，预测值，输出等。
  - scikit-learn对分析数据的四个关键的需求
    - 特征和结果标签**分别存储**在两个对象中
    - 特征和结果标签必须是**数字的**，如果是类别，需要将其编码到数字
    - 特征和结果便签的对象格式必须是numpy的**ndarray类**。由于pandas基于numpy构建，因此X和y可以使pandas的dataframe和series。
    - 特征和结果标签的维度必须是特定的。特征为2维，行是特定样本，列表示样本特征，结果标签为1维，表示其列别或真实数值，分别对应分类和回归问题

在sklearn中，通常将存储特征的对象命名为X，存储结果标签的对象定义为y。X一般大写，表示为矩阵。y一般小写，表示为向量。

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
  - 如何为监督学习任务选择模型，讲述了两个办法：
    - **在整个数据集上训练和测试模型**，训练和测试的数据相同。
      - 通过该方法比较了逻辑回归、knn-5，knn-1。其中knn-1 accuracy竟然为1
      - 通过knn算法的定义就可以很容易知道k=1时，在相同数据上都是确定的。
    - **将数据集分离成训练集和测试集**：实现方法采用sklearn提供的[train_test_split](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html#sklearn.model_selection.train_test_split)函数。代码实现如下：
    ```
    from sklearn.cross_validation import train_test_split
    X_train, X_test, y-train, y_test = train_test_split(X, y, test_size=0.4)
    # test_size：如果是0-1.0间的float，表示测试集所占比例;如果是整数表示测试样本数目，默认值为0.25
    # train_size与test_size相同
    # random_state：如该指定某个int，那么每次调用得到的随机数相同，也就是说，每个样本被分配为训练还是测试是不变的。
    # shuffle：分离前是否需要混洗，重新排序数据
    # stratify:是否按照类别或某个标签分类
    ```
    -
  - 如何计算分类模型的accuracy:
    - 通过sklearn.metrics包中的accuracy_score(y, y_pred)函数。
    ```
    from sklearn import metrics
    acc = metrics.accuracy_sore(y,y_pred)
    ```
  - 如何为模型调试最优的参数
    - 训练精确度：与模型复杂度成正比
    - 测试精确定：会受到模型复杂度的不利影响，模型过于复杂或过于简单，都会使得测试精确度降低。
    - 通常，最优参数也就是模型复杂度更好合适，不是过于复杂或过于简单。
    - **重要经验**：画出测试精确度与模型复杂度曲线，有助于分析参数
    - **重要经验**：当选定模型和最优参数后，用真个数据集重新训练模型是非常必要的。
  - 如何在样本之外，评估模型近似性能


### 6. 数据科学处理流程: pandas, seaborn, scikit-learn ([video](https://www.youtube.com/watch?v=3ZWuPVWq7p4&list=PL5-da3qGB5ICeMbQuqbbCOQWcS6OYBr5A&index=6), [notebook](06_linear_regression.ipynb), [blog post](http://blog.kaggle.com/2015/05/28/scikit-learn-video-6-linear-regression-plus-pandas-seaborn/))
#### 主要内容
  - 如何用pandas读取数据
    - 利用read_csv的index_col参数，可以将数据中的指定列作为dataframe的index。例如index_col=0
  - 如何用seaborn对数据进行可视化
    - seaborn基于matplotlib构建的数据可视化库
    ```
    import seaborn as sns
    %matplotlib inline
    #用散点图可视化特征和结果
    sns.pairplot(data, x_vars=['TV','Radio', 'Newspaper'], y_vars='sales',
                size=7, aspect=0.7, king='reg')
    # 输出图中直线表示拟合直线，蓝色区域表示95%的置信区间，通过图片可以直观看到数据符合线性特性，因此线性回归模型是一个好的可选模型
    ```
  - 线性回归以及其如何工作     
  $$ y = \beta_0 + \beta_1x_1 + \beta_2x_2 + ... + \beta_nx_n $$
    - 线性回归优缺点
      - 优点：运行速度快，并且随着数据规模花销线性增长;
      - 优点：无需调参（无超参数）
      - 优点：模型可解释性（interpretive）强，模型运行和预测机制容易理解
      - 优点：模型研究时间非常长，而且模型应用研究非常多
      - 缺点：相对于其他模型，线性回归模型假设特征与预测是线性关系，复杂度较低，然而现实中，很多都呈现非线性的关系，因此其预测准确率较差先`
    - 模型的训练采用最小二乘准则，也就是说让预测值与实际值之间差值平方和最小。
    - 线性回归对象中用`intercep_`属性存储偏差值`bias`，用`coef_`属性存储权重系数，其顺序与特征顺序一致。
  - scikit-learn中训练和理解线性回归模型(更详细代码或尝试运行，可下载讲义)
  ```
  # import model
  from sklearn.linear_model import LinearRegression
  # instantiate
  linreg = LinearRegression()
  # fit the model to the training data (learn the coefficients)
  linreg.fit(X_train, y_train)
  # print the intercept and coefficients
  print(linreg.intercept_)
  print(linreg.coef_)
  # pair the feature names with the coefficients
  list(zip(feature_cols, linreg.coef_))
  # make predictions on the testing set
  y_pred = linreg.predict(X_test)  
  ```
  - 回归问题的评估指标:
    - **平均绝对误差， Mean Absolute Error** (MAE) 是误差绝对值的均值:
      $$\frac 1n\sum_{i=1}^n|y_i-\hat{y}_i|$$
    - **均方误差，Mean Squared Error** (MSE)误差平方的均值:
      $$\frac 1n\sum_{i=1}^n(y_i-\hat{y}_i)^2$$
    - **均方根误差， Root Mean Squared Error** (RMSE) 均方误差的平方根:
      $$\sqrt{\frac 1n\sum_{i=1}^n(y_i-\hat{y}_i)^2}$$

    - sklearn代码：
    
      ```
      # calculate MAE using scikit-learn
      from sklearn import metrics
      print(metrics.mean_absolute_error(true, pred))

      # calculate MSE using scikit-learn
      print(metrics.mean_squared_error(true, pred))

      # calculate RMSE using scikit-learn
      print(np.sqrt(metrics.mean_squared_error(true, pred)))
      ```
    - 三者比较：
      - **MAE** 因为表示误差值的平均，直观并且容易理解。
      - **MSE** 比平均绝对误差用的更多，因为在MSE中，误差越大的数据影响也就越大，MSE也就会变大，换个角度来说，MSE会惩罚误差更大的数据。
      - **RMSE** 比MSE用的多，因为RMSE与y值单位相同，更直观理解误差。
  - 模型中的特征选择
    - 在课程实例中，发现线性模型中一个特征值的权重非常小，意味着该特征与预测值关联不大，因此可以从输入中将该特征删除，重新训练线性模型，从而发现模型的RMSE更小。
    - 根据线性模型的系数去分析特征与输出的关联性，从而进行特征选取也是一个不错的方法。

说明：**Association和causation(相关性和因果性)**。机器学习关注与数据相关性而不是因果性。


### 7. 交叉验证、参数调试、模型选择和特征选择 ([video](https://www.youtube.com/watch?v=6dbrR-WymjI&list=PL5-da3qGB5ICeMbQuqbbCOQWcS6OYBr5A&index=7), [notebook](07_cross_validation.ipynb), [blog post](http://blog.kaggle.com/2015/06/29/scikit-learn-video-7-optimizing-your-model-with-cross-validation/))
#### 主要内容
  - 评估模型中，训练和测试分离的缺点
    - 不同`random_state`竟然导致不同的准确性，当`random_state`为`5, 4, 3, 2, 1 `，准确性分别为`0.9473, 0.9736, 0.9473, 1, 1`。random_state参数只控制记录分配到测试集或训练集的随机性。不同的random_state值，测试集和训练集的数据不同，但不改变各自的数量。
    - 因此，训练和测试分离的缺点是，不同的随机分离策略，模型的准确性不同。也就是所谓测试准确性的高方差评估(high variance estimate)。

    ```
    # use train/test split with different random_state values
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=4)

    # check classification accuracy of KNN with K=5
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    print(metrics.accuracy_score(y_test, y_pred))
    ```
  - k-折交叉验证的步骤：
    - 将数据集分为k个数目相同的子集， **注意：分拆只进行一次，然后k次循环遍历不同的子集，每个数据都做了一次测试集。**
    - 将其中1个作为测试集，其余部分作为训练集
    - 利用上述训练和测试集，训练模型，并计算测试精确度
    - 将其他子集作为测试及，然后重复上面两个步骤，重复k次。使得每一个子集都做一次测试集
    - 用上述k个测试精确度的均值作为评估模型的精确度

  ```
    # simulate splitting a dataset of 25 observations into 5 folds
    from sklearn.cross_validation import KFold
    # 数据集中包含25个数据，分为5折，并且不对数据顺序进行调整
    kf = KFold(25, n_folds=5, shuffle=False)

    # print the contents of each training and testing set
    print('{} {:^61} {}'.format('Iteration', 'Training set observations', 'Testing set observations'))
    for iteration, data in enumerate(kf, start=1):
    print('{:^9} {} {:^25}'.format(iteration, data[0], data[1]))
  ```
  - k折交叉验证与训练和测试分离的不同：
    - k折交叉验证，更准确地评估样本外的准确性;更有效地利用数据（每个样本都用于训练和测试）
    - 训练和测试分离：速度更快，无需循环k次;更直观地反映测试过程，更方便去探索测试细节，以便于更好第调参
  - k-折交叉验证 **两个重要经验**：
    - k通常取值为10
    - 分类中，采用分层抽样(**stratified sampling**)，使每个子集中不同类别数据比例相同。sklearn的`cross_val_score`实现了该功能。交叉验证的示例如下，实例是关于KNN模型，代码对参数K进行从1到31的遍历搜索，针对每个k值，利用`cross_val_score`获得10个10-折交叉验证准确性，并将其均值存储到到`k_scores`列表。注意，在使用`cross_val_score`中无需手工分离训练和测试集。
    ```
    from sklearn.cross_validation import cross_val_score

    # search for an optimal value of K for KNN
    k_range = list(range(1, 31))
    k_scores = []
    for k in k_range:
      knn = KNeighborsClassifier(n_neighbors=k)
      scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
      k_scores.append(scores.mean())
    print(k_scores)
    ```

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
