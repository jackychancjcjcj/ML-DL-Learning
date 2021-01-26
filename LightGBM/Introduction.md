# LightGBM
LightGBM （Light Gradient Boosting Machine）[github开源](https://github.com/Microsoft/LightGBM)是一个实现GBDT算法的框架，支持高效率的并行训练，并且具有以下优点：  
● 更快的训练速度  
● 更低的内存消耗  
● 更好的准确率  
● 分布式支持，可以快速处理海量数据  
从下图实验数据可以看出，在Higgs数据集上LightGBM比XGBoost快将近10倍，内存占用率大约为XGBoost的1/6，并且准确率也有提升。在其他数据集上也可以观察到相似的结论。    
![`训练速度`](https://github.com/jackychancjcjcj/ML-DL-Learning/blob/master/lightgbmV1.jpg)
![`内存消耗`](https://github.com/jackychancjcjcj/ML-DL-Learning/blob/master/lightgbmV2.jpg)
![`准确率`](https://github.com/jackychancjcjcj/ML-DL-Learning/blob/master/lightgbmV3.jpg)
## 提出动机
对于常用的机器学习，都可以采用mini-batch的方法，训练数据的大小不会受到内存的限制。  
但对于gbdt类算法来说，每次迭代都需要读取全部数据，而如果把数据都装入内存，内存大小就会限制数据大小，尤其在工业级海量数据下这是不可能的。而如果都装入硬盘，则I/O将会消耗大量的时间。  
### XGBoost是如何工作的？
目前已有的GBDT工具基本都是基于预排序的方法（pre-sorted）的决策树算法(如xgboost)。这种构建决策树的算法基本思想是：  
* 对所有特征都按照特征的数值进行预排序。  
* 在遍历分割点的时候用O(#data)的代价找到一个特征上的最好分割点。  
* 找到一个特征的分割点后，将数据分裂成左右子节点。  
### 这样的预排序算法的优点是能精确地找到分割点，缺点也很明显：  
* 空间消耗大。这样的算法需要保存数据的特征值，还保存了特征排序的结果（例如排序后的索引，为了后续快速的计算分割点），这里需要消耗训练数据两倍的内存。  
* 时间上也有较大的开销，在遍历每一个分割点的时候，都需要进行分裂增益的计算，消耗的代价大。  
* 对cache优化不友好。在预排序后，特征对梯度的访问是一种随机访问，并且不同的特征访问的顺序不一样，无法对cache进行优化。同时，在每一层长树的时候，需要随机访问一个行索引到叶子索引的数组，并且不同特征访问的顺序也不一样，也会造成较大的cache miss。
### LightGBM在哪些地方进行了优化？
基于Histogram的决策树算法带深度限制的Leaf-wise的叶子生长策略直方图做差加速直接支持类别特征(Categorical Feature)Cache命中率优化基于直方图的稀疏特征优化多线程优化下面主要介绍`Histogram算法`、`带深度限制的Leaf-wise的叶子生长策略`和`直方图做差加速`。  
#### Histogram算法
直方图算法的基本思想是先把连续的浮点特征值离散化成k个整数，同时构造一个宽度为k的直方图。在遍历数据的时候，根据离散化后的值作为索引在直方图中累积统计量，当遍历一次数据后，直方图累积了需要的统计量，然后根据直方图的离散值，遍历寻找最优的分割点。  
![](https://github.com/jackychancjcjcj/ML-DL-Learning/blob/master/histogram%E7%AE%97%E6%B3%95.jpg)
使用直方图算法有很多优点。首先，最明显就是内存消耗的降低，直方图算法不仅不需要额外存储预排序的结果，而且可以只保存特征离散化后的值，而这个值一般用8位整型存储就足够了，内存消耗可以降低为原来的1/8  
![](https://github.com/jackychancjcjcj/ML-DL-Learning/blob/master/histogram%E7%AE%97%E6%B3%952.jpg)
然后在计算上的代价也大幅降低，预排序算法每遍历一个特征值就需要计算一次分裂的增益，而直方图算法只需要计算k次（k可以认为是常数），时间复杂度从O(#data*#feature)优化到O(k*#features)。  
当然，Histogram算法并不是完美的。由于特征被离散化后，找到的并不是很精确的分割点，所以会对结果产生影响。但在不同的数据集上的结果表明，离散化的分割点对最终的精度影响并不是很大，甚至有时候会更好一点。原因是决策树本来就是弱模型，分割点是不是精确并不是太重要；较粗的分割点也有正则化的效果，可以有效地防止过拟合；即使单棵树的训练误差比精确分割的算法稍大，但在梯度提升（Gradient Boosting）的框架下没有太大的影响。
#### 带深度限制的Leaf-wise的叶子生长策略
在Histogram算法之上，LightGBM进行进一步的优化。首先它抛弃了大多数GBDT工具使用的按层生长 (level-wise)的决策树生长策略，而使用了带有深度限制的按叶子生长 (leaf-wise)算法。Level-wise过一次数据可以同时分裂同一层的叶子，容易进行多线程优化，也好控制模型复杂度，不容易过拟合。但实际上Level-wise是一种低效的算法，因为它不加区分的对待同一层的叶子，带来了很多没必要的开销，因为实际上很多叶子的分裂增益较低，没必要进行搜索和分裂。  
![](https://github.com/jackychancjcjcj/ML-DL-Learning/blob/master/leaf-wise1.jpg)
Leaf-wise则是一种更为高效的策略，每次从当前所有叶子中，找到分裂增益最大的一个叶子，然后分裂，如此循环。因此同Level-wise相比，在分裂次数相同的情况下，Leaf-wise可以降低更多的误差，得到更好的精度。Leaf-wise的缺点是可能会长出比较深的决策树，产生过拟合。因此LightGBM在Leaf-wise之上增加了一个最大深度的限制，在保证高效率的同时防止过拟合。  
![](https://github.com/jackychancjcjcj/ML-DL-Learning/blob/master/leaf-wise2.jpg)
#### 直方图差加速
LightGBM另一个优化是Histogram（直方图）做差加速。一个容易观察到的现象：一个叶子的直方图可以由它的父亲节点的直方图与它兄弟的直方图做差得到。通常构造直方图，需要遍历该叶子上的所有数据，但直方图做差仅需遍历直方图的k个桶。利用这个方法，LightGBM可以在构造一个叶子的直方图后，可以用非常微小的代价得到它兄弟叶子的直方图，在速度上可以提升一倍。  
![](https://github.com/jackychancjcjcj/ML-DL-Learning/blob/master/%E7%9B%B4%E6%96%B9%E5%9B%BE%E5%B7%AE%E5%8A%A0%E9%80%9F.jpg)


## 参考文章
https://www.msra.cn/zh-cn/news/features/lightgbm-20170105  
https://zhuanlan.zhihu.com/p/24498293  
[kaggle神器lightGBM最全解读](https://mp.weixin.qq.com/s/64xfT9WIgF3yEExpSxyshQ)  
