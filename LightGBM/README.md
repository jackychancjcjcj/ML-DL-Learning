# LightGBM 参数调整
    pip install lightgbm
## lightgbm.LGBMClassifier
```python
__init__(boosting_type='gbdt', num_leaves=31, max_depth=- 1, learning_rate=0.1, n_estimators=100, subsample_for_bin=200000, objective=None, class_weight=None, min_split_gain=0.0, min_child_weight=0.001, min_child_samples=20, subsample=1.0, subsample_freq=0, colsample_bytree=1.0, reg_alpha=0.0, reg_lambda=0.0, random_state=None, n_jobs=- 1, silent=True, importance_type='split', **kwargs)
```
### Parameters
    boosting_type: string, default='gbdt'
        gbdt: 传统的梯度提升树  
        dart: [`Dropouts meet Multiple Additive Regression Trees`](https://arxiv.org/abs/1505.01866)  
        goss: Gradient-based One-Side Sampling (基于梯度的单侧采样)  
        rf: Random Forest  
    num_leaves: int, default=31  
        每个弱学习器拥有的叶子的最大数量。较大的值增加了训练集的精确度，也增加了过拟合。根据文档，简单的方法是小于num_leaves=2^(max_depth)。  
    max_depth: int, default=-1  
        控制每颗经过训练的树的最大深度，-1代表无限制。  
    learning_rate: float, default=0.1  
        学习率，一般是0.05-0.2左右。 
    n_estimators: int, default=100  
        控制弱学习器数量  
    subsample_for_bin: int, default=200000  
        一般不调  
    objective: string, default=None  
        default: 'regression' for LGBMRegressor,'binary' or 'multiclass' for LGBMClassifier  
    class_weight: dict, default=None  
    min_split_gain: float, default=0  
    min_child_weight: float, default=1e-3  
    min_child_samples: int, default=20  
    subsample: float,default=1  
    subsample_freq: int, default=0  
        0 表示不用  
    colsample_bytree: float, default=1  
    reg_alpha: float, default=0
        L1正则  
    reg_lambda: float, default=0  
        L2正则  
    random_stage: int, default=None  
    n_jobs: int, default=-1  
        线程数  
    silent: bool, default=True  
        打印信息与否  
### Methods
    `__init__`: 初始化一个model  
    `fit`: 训练  
    `get_params`: 获取参数  
    `predict`: 预测  
    `predict_proba`: 获得每个预测种类的可能性  
    `set_params`: 设置参数  
### Attributes
    best_iteration_：The best iteration of fitted model if `early_stopping_rounds` has been specified.  
    best_score_: The best score of fitted model  
    booster_: The underlying Booster of this model  
    classes_: The class label array  
    evals_result_: The evaluation results if early_stopping_rounds has been specified.  
    feature_importances_: The feature importances (the higher, the more important).  
    feature_name_:  The names of features.  
    n_classes_:  The number of classes.  
    n_features_:  The number of features of fitted model.
### lightgbm.LGBMClassifier.fit
```python
fit(X, y, sample_weight=None, init_score=None, eval_set=None, eval_names=None, eval_sample_weight=None, eval_class_weight=None, eval_init_score=None, eval_metric=None, early_stopping_rounds=None, verbose=True, feature_name='auto', categorical_feature='auto', callbacks=None, init_model=None)
```
#### Parameters
    X: 训练集  
    y: traget结果  
    sample_weight: weight of training data  
    init_score: default=None  
    eval_set: a list of (X,y) 用作验证集  
    eval_names: default=None  
    eval_sample_weight: weights of eval data  
    eval_metric: dict, default=None  
        {'auc','l2','rmse','f1'}  
    early_stopping_rounds: int, default=None  
        模型会一直训练直到验证分数停止提升，设置这个参数使得若验证分数一直提升的话可以提前结束训练。  
    verbose: bool or int, default=True  
        If True, the eval metric on the eval set is printed at each boosting stage. If int, the eval metric on the eval set is printed at every verbose boosting stage.  
### lightgbm.LGBMClassifier.predict
```python
predict(X, raw_score=False, start_iteration=0, num_iteration=None, pred_leaf=False, pred_contrib=False, **kwargs)
```
#### Parameters
    X: 预测数据集  
    raw_score: bool, default=None
        whether to predict raw scores  
    start_iteration: int, default=0  
        start index of the iteration to predict, 0代表从头开始。  
    num_iteration: int, default=0
        迭代次数，0代表用最优的迭代次数  
    pred_leaf: bool, default=None  
        whether to predict leaf index  
    pred_contrib: bool, default=None  
        whether to predict feature contributions  
## 调参策略 
* step 1.确定学习率和估计器数目  
    das
        
