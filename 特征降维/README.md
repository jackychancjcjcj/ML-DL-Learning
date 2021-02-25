# 目录
* [过滤法(filter)](#1)
  * [VarianceThreshold](#1.1)
  * [SelectKBest](#1.2) 
* [包装法(wrapper)](#2)
  * [RFE](#2.1)
* [嵌入法(embedded)](#3)
  * [SelectFromModel](#3.1)
* [线性降维](#4)
## <span id='1'>过滤法</span>
`过滤法`按照发散性或者相关性对各个特征进行评分，通过设定阈值或者待选择阈值的个数来选择特征。
### <span id='1.1'>VarianceThreshold</span>
计算特征方差，根据阈值选择特征。
```python
from sklearn.feature_selection import VarianceThreshold
from sklearn.datasets import load_iris
iris = load_iris()
# 参数threshold就是设定的阈值
VarianceThreshold(threshold=3).fit_transform(iris.data)
```
### <span id='1.2'>SelectKBest</span>
(1) 相关系数法：计算各个特征对于目标值的相关系数及相关系数的p值。
```python
from sklearn.feature_selection import SelectKBest
import numpy as np
from sklearn.datasets import load_iris
from scipy.stats import pearsonr
iris = load_iris()
# 参数k就是个数，第一函数就是评价函数
SelectKBest(
    lambda X,Y: np.array(list(map(lambda x:pearsonr(x,Y),X.T))).T[0],k=2
).fit_transform(iris.data,iris.target)
```
(2) 卡方检验：检验定性自变量与定性因变量的相关性。
 ```python
 from sklearn.feature_selection import SelectKBest
import numpy as np
from sklearn.datasets import load_iris
from sklearn.feature_selection import chi2
iris = load_iris()
# 参数k就是选择的特征个数，第一函数就是评价函数
SelectKBest(
    chi2,k=2
).fit_transform(iris.data,iris.target)
 ```
## <span id='2'>包装法</span>
根据目标函数（一般是预测评分）选择若干特征，或者排除若干特征。
### <span id='2.1'>RFE</span>
使用一个基模型进行多轮训练，每轮训练会消除若干权值系数特征，再基于新的数据集进行下一轮训练。
```python
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
iris = load_iris()
# 参数estimator是基模型，n_features_to_select是选择的特征个数
RFE(estimator=LogisticRegression(multi_class='auto',solver='lbfgs',max_iter=500),n_features_to_select=2).fit_transform(iris.data,iris.target)
```
## <span id='3'>嵌入法</span>
使用机器学习的算法和模型训练，得到特征重要性，根据这个来选择特征。
### <span id='3.1'>SelectFromModel</span>
基于模型的特征选择法，可以基于惩罚项或者基于树模型。
(1) 基于惩罚项：
```python
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
iris = load_iris()
SelectFromModel(LogisticRegression(penalty='l2',multi_class='auto',solver='lbfgs',C=0.1)).fit_transform(iris.data,iris.target)
```
(2) 基于树模型：
```python
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import GradientBoostingClassifier
iris = load_iris()
SelectFromModel(GradientBoostingClassifier()).fit_transform(iris.data,iris.target)
```
