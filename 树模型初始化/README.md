# 树模型初始化技巧
## 传统策略
```python
import pandas as pd
import numpy as np
import lightgbm as lgbm
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
##### 1.读取数据
train = pd.read_csv("...")
test = pd.read_csv("...")
##### 2.N折训练测试
cont_features = [col for col in train.columns if col.startswith("cont")]
len(cont_features) 
y = train["target"]
kf = KFold(n_splits=5, shuffle=True, random_state=1)
oof = np.zeros(len(train))
score_list = []
fold = 1
test_preds = []


for train_index, test_index in kf.split(train):
    X_train, X_val = train.iloc[train_index], train.iloc[test_index]
    y_train, y_val = y.iloc[train_index], y.iloc[test_index] 
    
    y_pred_list = []
    for seed in [1]:
        dtrain = lgbm.Dataset(X_train[cont_features], y_train)
        dvalid = lgbm.Dataset(X_val[cont_features], y_val)
        print(seed)
        params = {"objective": "regression",
              "metric": "rmse",
              "verbosity": -1,
              "boosting_type": "gbdt",
              "feature_fraction":0.5,
              "num_leaves": 200,
              "lambda_l1":2,
              "lambda_l2":2,
              "learning_rate":0.01,
              'min_child_samples': 50,
              "bagging_fraction":0.7,
              "bagging_freq":1}
        params["seed"] = seed
        model = lgbm.train(params,
                        dtrain,
                        valid_sets=[dtrain, dvalid],
                        verbose_eval=100,
                        num_boost_round=100000,
                        early_stopping_rounds=100
                    )
    
        y_pred_list.append(model.predict(X_val[cont_features]))
        test_preds.append(model.predict(test[cont_features])) 
        
    oof[test_index] = np.mean(y_pred_list,axis=0)    
    score = np.sqrt(mean_squared_error(y_val, oof[test_index]))
    score_list.append(score)
    print(f"RMSE Fold-{fold} : {score}")
    fold+=1

np.mean(score_list)
```
## Trick版本
大致思路：  
* 用一个较大的learning rate学习得到初始版本模型1；
* 用一个较小的learning rate在模型1上继续训练得到模型2；
```python
import pandas as pd
import numpy as np
import lightgbm as lgbm
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
##### 1.读取数据
train = pd.read_csv("...")
test = pd.read_csv("...")
##### 2.N折训练测试
cont_features = [col for col in train.columns if col.startswith("cont")]
len(cont_features) 
y = train["target"]
kf = KFold(n_splits=5, shuffle=True, random_state=1)
oof = np.zeros(len(train))
score_list = []
fold = 1
test_preds = []


for train_index, test_index in kf.split(train):
    X_train, X_val = train.iloc[train_index], train.iloc[test_index]
    y_train, y_val = y.iloc[train_index], y.iloc[test_index]
    
    

    X_train = X_train.abs()

    
    y_pred_list = []
    for seed in [1]:
        dtrain = lgbm.Dataset(X_train[cont_features], y_train)
        dvalid = lgbm.Dataset(X_val[cont_features], y_val)
        print(seed)
        params = {"objective": "regression",
              "metric": "rmse",
              "verbosity": -1,
              "boosting_type": "gbdt",
              "feature_fraction":0.5,
              "num_leaves": 200,
              "lambda_l1":2,
              "lambda_l2":2,
              "learning_rate":0.01,
              'min_child_samples': 50,
              "bagging_fraction":0.7,
              "bagging_freq":1}
        params["seed"] = seed
        model = lgbm.train(params,
                        dtrain,
                        valid_sets=[dtrain, dvalid],
                        verbose_eval=100,
                        num_boost_round=100000,
                        early_stopping_rounds=100
                    )
        
        
        ##### 3. 额外的策略
        dtrain = lgbm.Dataset(X_train[cont_features], y_train)
        dvalid = lgbm.Dataset(X_val[cont_features], y_val)
        params = {"objective": "regression",
              "metric": "rmse",
              "verbosity": -1,
              "boosting_type": "gbdt",
              "feature_fraction":0.5,
              "num_leaves": 300,
              "lambda_l1":2,
              "lambda_l2":2,
              "learning_rate":0.003,
              'min_child_samples': 50,
              "bagging_fraction":0.7,
              "bagging_freq":1}

        params["seed"] = seed
        model = lgbm.train(params,
                            dtrain,
                            valid_sets=[dtrain, dvalid],
                            verbose_eval=100,
                            num_boost_round=1000,
                           early_stopping_rounds=100,
                           init_model = model
                        )

    
    
        y_pred_list.append(model.predict(X_val[cont_features]))
        test_preds.append(model.predict(test[cont_features]))
    
   
    
    oof[test_index] = np.mean(y_pred_list,axis=0)    
    score = np.sqrt(mean_squared_error(y_val, oof[test_index]))
    score_list.append(score)
    print(f"RMSE Fold-{fold} : {score}")
    fold+=1

np.mean(score_list)

```
