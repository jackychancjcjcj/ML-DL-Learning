# 模型集成归纳
## 目录
* [1.对于结果的集成](#1)  
  * [1.1投票集成(voting ensemble)](#1.1)
  * [1.2算术平均数集成(Arithmetic mean based ensemble)](#1.2)
  * [1.3几何平均数集成(Geometric mean based ensemble)](#1.3)
  * [1.4线上结果加权集成(Online scroe based ensemble)](#1.4)
  * [1.5排序均值集成(Rank averaging ensemble)](#1.5)
  * [1.6log集成变种(log ensemble version2)](#1.6)
  * [1.7排序平均(rank averaging)](#1.7)
* [2.Stacking/Blending/Bagging](#2) 
## 参考
[竞赛集成CookBook](https://mp.weixin.qq.com/s?__biz=Mzk0NDE5Nzg1Ng==&mid=2247490302&idx=1&sn=cc850f781a7497ab6fad04e8b2f6e07c&chksm=c3290371f45e8a67a4abf37ef4f352ebea1567c0f59e8dcbf27e5c17ba81a90b70f37a41638d&mpshare=1&scene=1&srcid=0207hpIku2RhyAy3EzX80iM5&sharer_sharetime=1612693105638&sharer_shareid=9b869c9a24181fe91d7ddd3f39c6511b&version=3.1.1.3006&platform=win#rd)
## <span id='1'>1.对于结果的集成</span>
## <span id='1.1'>1.1投票集成(voting ensemble)</span>
输入:多个不同分类器的分类结果;  
输出:最终的集成结果  
基本步骤:  
统计每个样本每个预测结果(常见于分类问题)出现的次数;  
将每个样本出现的次数最多的那一个(众数)作为我们最终的集成结果.   
## <span id='1.2'>1.2算术平均数集成(Arithmetic mean based ensemble)</span>
```python
sub['final_result'] = (sub['result_1'] + sub['result_2'] + sub['result_3'])/3
```
## <span id='1.3'>1.3几何平均数集成(Geometric mean based ensemble)</span>
```python
sub['final_result'] = (sub['result_1'] * sub['result_2'] * sub['result_3'])**(1/3)
```
## <span id='1.7'>1.7排序平均(rank averaging)</span>
```python
sub1 = pd.read_csv('../model/xgb_2020.csv')  # xgb
sub2 = pd.read_csv(open(r'../model/cat_2020.csv'))  # cat
sublist = [sub1, sub2]
fusion = sub2.copy()
fusion.isDefault = np.sqrt(sub1.isDefault.rank() * sub2.isDefault.rank())
```
## <span id='2'>2.Stacking/Blending/Bagging</span>
[Stacking](https://github.com/jackychancjcjcj/ML-DL-Learning/tree/master/Stacking)
