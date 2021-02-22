# 机器学习四个常用的超参数调试方法

* [传统手工调参](#1)
* [网格搜索](#2)
* [随即搜索](#3)
* [贝叶斯搜索](#4)

## <span id='1'>传统手工搜索</span>
```python
#importing required libraries
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold , cross_val_score
from sklearn.datasets import load_wine

wine = load_wine()
X = wine.data
y = wine.target

#splitting the data into train and test set
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state = 14)

#declaring parameters grid
k_value = list(range(2,11))
algorithm = ['auto','ball_tree','kd_tree','brute']
scores = []
best_comb = []
kfold = KFold(n_splits=5)

#hyperparameter tunning
for algo in algorithm:
  for k in k_value:
    knn = KNeighborsClassifier(n_neighbors=k,algorithm=algo)
    results = cross_val_score(knn,X_train,y_train,cv = kfold)

    print(f'Score:{round(results.mean(),4)} with algo = {algo} , K = {k}')
    scores.append(results.mean())
    best_comb.append((k,algo))

best_param = best_comb[scores.index(max(scores))]
print(f'\nThe Best Score : {max(scores)}')
print(f"['algorithm': {best_param[1]} ,'n_neighbors': {best_param[0]}]")
```
可以看到的是  
1. 没办法确保得到最佳的参数组合。  
2. 这是一个不断试错的过程，所以非常耗时。

## <span id='2'>网格搜索</span>
网格搜索是一种基本的超参数调优技术，考虑上面的例子，其中两个超参数k_value =[2,3,4,5,6,7,8,9,10] & algorithm =[' auto '， ' ball_tree '， ' kd_tree '， ' brute ']，在这个例子中，它总共构建了9*4 = 36不同的模型。
```python
from sklearn.model_selection import GridSearchCV

knn = KNeighborsClassifier()
grid_param = { 'n_neighbors' : list(range(2,11)) , 
              'algorithm' : ['auto','ball_tree','kd_tree','brute'] }
              
grid = GridSearchCV(knn,grid_param,cv = 5)
grid.fit(X_train,y_train)

#best parameter combination
grid.best_params_

#Score achieved with best parameter combination
grid.best_score_

#all combinations of hyperparameters
grid.cv_results_['params']

#average scores of cross-validation
grid.cv_results_['mean_test_score']
```
缺点是尝试了每一个超参数组合，通过交叉验证得分选择最佳组合，速度很慢。  

### 实际应用代码
```python
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
def rf_params_gridsearch(model,train_x,train_y,label_split=None):
    train_data,test_data,train_target,test_target = train_test_split(train_x,train_y,test_size=0.2,random_state=2020)
    parameters = {'min_samples_split' : range(1,10,1)}
    n_splits=5
    sk = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=2020)
    clf = GridSearchCV(model,parameters,cv=sk.split(train_x,train_y),verbose=2,scoring='f1')
    clf.fit(train_x,train_y)
    print('rf:')
    print(clf.best_score_ )
    print(clf.best_params_)
rf = RandomForestClassifier(oob_score=True, random_state=2020,
            n_estimators= 50,max_depth=13,min_samples_split=5)
rf_params_gridsearch(rf,train_data,kind)
```

## <span id='3'>随机搜索</span>
使用随机搜索代替网络搜索动机是，在许多情况下，所有的超参数可能不是同等重要的。随机搜索从超参数空间中随机选择参数组合，有n_iter给定的固定迭代次数的情况下选择。
```python
from sklearn.model_selection import RandomizedSearchCV

knn = KNeighborsClassifier()

grid_param = { 'n_neighbors' : list(range(2,11)) , 
              'algorithm' : ['auto','ball_tree','kd_tree','brute'] }

rand_ser = RandomizedSearchCV(knn,grid_param,n_iter=10)
rand_ser.fit(X_train,y_train)

#best parameter combination
rand_ser.best_params_

#score achieved with best parameter combination
rand_ser.best_score_

#all combinations of hyperparameters
rand_ser.cv_results_['params']

#average scores of cross-validation
rand_ser.cv_results_['mean_test_score']
```
缺点是随机搜索不能保证给出最好的参数组合

## <span id='4'>贝叶斯搜索</span>
贝叶斯优化属于一类优化算法，称为基于序列模型的优化算法，这些算法使用先前对损失f的观察结果，以确定下一个（最优）点来抽样f。  
1. 使用先前评估的点X1*:n*，计算损失f的后验期望。  
2. 在新的点X的抽样损失f，从而最大化f的期望的某些方法。该方法指定f域的哪些区域最适于抽样。  
需要利用`scikit-optimization`的`BayesSearchCV`使用
    Installation: pip install scikit-optimize
```python
from skopt import BayesSearchCV

import warnings
warnings.filterwarnings("ignore")

# parameter ranges are specified by one of below
from skopt.space import Real, Categorical, Integer

knn = KNeighborsClassifier()
#defining hyper-parameter grid
grid_param = { 'n_neighbors' : list(range(2,11)) , 
              'algorithm' : ['auto','ball_tree','kd_tree','brute'] }

#initializing Bayesian Search
Bayes = BayesSearchCV(knn , grid_param , n_iter=30 , random_state=14)
Bayes.fit(X_train,y_train)

#best parameter combination
Bayes.best_params_

#score achieved with best parameter combination
Bayes.best_score_

#all combinations of hyperparameters
Bayes.cv_results_['params']

#average scores of cross-validation
Bayes.cv_results_['mean_test_score']
```
缺点是要在2维或3维的搜索空间中得到一个好的代理曲面需要十几个样本，增加搜索空间的维数需要更多的样本。  

## 参考
[https://mp.weixin.qq.com/s/V3HzYBlOsMo3C_Hf4r2OqA](https://mp.weixin.qq.com/s/V3HzYBlOsMo3C_Hf4r2OqA)
