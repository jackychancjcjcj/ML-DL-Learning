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
基于Histogram的决策树算法带深度限制的Leaf-wise的叶子生长策略直方图做差加速直接支持类别特征(Categorical Feature)Cache命中率优化基于直方图的稀疏特征优化多线程优化下面主要介绍Histogram算法、带深度限制的Leaf-wise的叶子生长策略和直方图做差加速。




## 参考文章
https://www.msra.cn/zh-cn/news/features/lightgbm-20170105  
https://zhuanlan.zhihu.com/p/24498293
