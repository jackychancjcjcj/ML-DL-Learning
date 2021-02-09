# 模型集成归纳
## 目录
## 参考
* [1.对于结果的集成](#1)  
** [1.1投票集成voting ensemble](#1.1)
* [2.Stacking/Blending/Bagging](#2) 
[竞赛集成CookBook](https://mp.weixin.qq.com/s?__biz=Mzk0NDE5Nzg1Ng==&mid=2247490302&idx=1&sn=cc850f781a7497ab6fad04e8b2f6e07c&chksm=c3290371f45e8a67a4abf37ef4f352ebea1567c0f59e8dcbf27e5c17ba81a90b70f37a41638d&mpshare=1&scene=1&srcid=0207hpIku2RhyAy3EzX80iM5&sharer_sharetime=1612693105638&sharer_shareid=9b869c9a24181fe91d7ddd3f39c6511b&version=3.1.1.3006&platform=win#rd)
## <span id='1'>1.对于结果的集成</span>
## <span id='1.1'>1.1投票集成(voting ensemble)</span>
输入:多个不同分类器的分类结果;  
输出:最终的集成结果  
基本步骤:  
统计每个样本每个预测结果(常见于分类问题)出现的次数;  
将每个样本出现的次数最多的那一个(众数)作为我们最终的集成结果.   
```python

```
## 2.Stacking/Blending/Bagging
## 
