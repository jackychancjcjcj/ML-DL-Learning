# 斩获无数金牌的技能：GBDT+LR/FFM/DNN混合训练
## 背景
在数据竞赛圈流传着一种秘法，人称GBDT+LR/FFM/DNN的策略，大家都知道GBDT类的模型在数值类特征的处理上有着较强的能力，而NN等模型到目前为止都很难找到一种方案能在数值特征的处理上达到GBDT类模型的效果，那怎么办呢?  
一群数据竞赛大神，包括曾经的KDD冠军台湾大学队伍，kaggle的竞赛狂人Faron等在之前的比赛中对于此类秘技的使用出神入化，也拿了无数的金牌绿牌。后来FaceBook的论文也谈到了此类方案在实际业务中的巨大效果，那究竟是怎么做的呢？  
本文就结合代码来简单阐述此类方案的整体流程，大家今后的数据赛中也可以收藏一下。  
Facebook的GBDT+LR策略的大体思路如下图所示，先使用GBDT类的模型训练预测得到叶子节点的预测结果，然后基于叶子结果进行特征的变换，并基于变换后的叶子特征进行LR/FFM/DNN等模型训练预测。  
![逻辑图](https://github.com/jackychancjcjcj/ML-DL-Learning/blob/master/GBDTandLR.png)
## 代码
### 训练集测试集划分
```python
train = data[data['Label'] != -1]
target = train.pop('Label')
test = data[data['Label'] == -1]
test.drop(['Label'], axis = 1, inplace = True)
```
### 训练LGB模型
```python
# 划分数据集
print('划分数据集...')
x_train, x_val, y_train, y_val = train_test_split(train, target, test_size = 0.2, random_state = 2018)

print('开始训练gbdt..')
gbm = lgb.LGBMRegressor(objective='binary',
                        subsample= 0.8,
                        min_child_weight= 0.5,
                        colsample_bytree= 0.7,
                        num_leaves=100,
                        max_depth = 12,
                        learning_rate=0.05,
                        n_estimators=10,
                        )
                        
gbm.fit(x_train, y_train, eval_set = [(x_train, y_train), (x_val, y_val)],
        eval_names = ['train', 'val'], eval_metric = 'binary_logloss', )
```
### 树节点one-hot编码
```python
model               = gbm.booster_
gbdt_feats_train    = model.predict(train, pred_leaf = True) # pred_leaf = True的话，输出的就是一个行是样本数，列是样本所在树节点的矩阵。（列数等于树的数目）
gbdt_feats_test     = model.predict(test, pred_leaf = True)
gbdt_feats_name     = ['gbdt_leaf_' + str(i) for i in range(gbdt_feats_train.shape[1])] # shape[1]也就是树的数目
df_train_gbdt_feats = pd.DataFrame(gbdt_feats_train, columns = gbdt_feats_name) 
df_test_gbdt_feats  = pd.DataFrame(gbdt_feats_test, columns = gbdt_feats_name)
 # 以上可以看到是把树的特征加进去了
    
train = pd.concat([train, df_train_gbdt_feats], axis = 1)
test  = pd.concat([test, df_test_gbdt_feats], axis = 1)
train_len = train.shape[0]
data      = pd.concat([train, test])
del train
del test
gc.collect()
 
 
print('one-hot features for leaf node')
for col in gbdt_feats_name:
    print('feature:', col)
    onehot_feats = pd.get_dummies(data[col], prefix = col)
    data.drop([col], axis = 1, inplace = True)
    data = pd.concat([data, onehot_feats], axis = 1)
print('one-hot结束')

train = data[: train_len]
test = data[train_len:]
del data
gc.collect() 
```
### 训练LR模型并输出
```python
x_train, x_val, y_train, y_val = train_test_split(train, target, test_size = 0.3, random_state = 2018)
lr = LogisticRegression()
lr.fit(x_train, y_train) 
y_pred = lr.predict_proba(test)[:, 1]
```
