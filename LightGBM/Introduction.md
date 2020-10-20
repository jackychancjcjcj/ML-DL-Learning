# LightGBM
LightGBM （Light Gradient Boosting Machine）[github开源](https://github.com/Microsoft/LightGBM)是一个实现GBDT算法的框架，支持高效率的并行训练，并且具有以下优点：  
● 更快的训练速度  
● 更低的内存消耗  
● 更好的准确率  
● 分布式支持，可以快速处理海量数据  
从下图实验数据可以看出，在Higgs数据集上LightGBM比XGBoost快将近10倍，内存占用率大约为XGBoost的1/6，并且准确率也有提升。在其他数据集上也可以观察到相似的结论。    
！[`训练速度`](https://github.com/jackychancjcjcj/ML-DL-Learning/blob/master/lightgbmV1.jpg)
## 提出动机
