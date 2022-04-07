## 初始化环境
```python
def seed_everything(seed=2020):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
```
## 阈值
```python
def find_best_threshold(y_valid, oof_prob):
    best_f2 = 0

    for th in tqdm([i / 1000 for i in range(50, 200)]):
        oof_prob_copy = oof_prob.copy()
        oof_prob_copy[oof_prob_copy >= th] = 1
        oof_prob_copy[oof_prob_copy < th] = 0

        recall = recall_score(y_valid, oof_prob_copy)
        precision = precision_score(y_valid, oof_prob_copy)
        f2 = 5 * recall * precision / (4 * precision + recall)

        if f2 > best_f2:
            best_th = th
            best_f2 = f2

        gc.collect()

    return best_th, best_f2

best_th, best_f2 = find_best_threshold(y_valid, oof_prob)
print(best_th, best_f2)
```
## 标准代码
TIPS:  
1.LGB可以自己处理na，但我们也可以先处理na再给lgb  
2.lgb可以处理categorical_features,做法：
```python
for i in cate_feat:
        data_df[i] = data_df[i].astype('category')
params = {
    'categorical_feature':cate_feat
}
```
## lgb sklearn
```python
from lightgbm.sklearn import LGBMClassifier
from sklearn.metrics import f1_score, roc_auc_score
train_df = df[df['label'].isna() == False].reset_index(drop=True)
test_df = df[df['label'].isna() == True].reset_index(drop=True)
display(train_df.shape, test_df.shape)
cols = [col for col in train_df.columns if col not in ['id', 'label']]

oof = np.zeros(train_df.shape[0])
df_importance_list = []
train_df['prob'] = 0
test_df['prob'] = 0
clf = LGBMClassifier(
    boosting='gbdt',
    objective='binary',
    learning_rate=0.05,
    n_estimators=10000,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=1023,
    reg_alpha=.3,
    reg_lambda=.3,
    min_split_gain=.01,
    min_child_weight=2,
    metric=None
)

val_aucs = []
seeds = [1023, 2048, 2098]
for seed in seeds:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for i, (trn_idx, val_idx) in enumerate(skf.split(train_df, train_df['label'])):
        print('--------------------- {} fold ---------------------'.format(i))
        trn_x, trn_y = train_df[cols].iloc[trn_idx].reset_index(drop=True), train_df['label'].values[trn_idx]
        val_x, val_y = train_df[cols].iloc[val_idx].reset_index(drop=True), train_df['label'].values[val_idx]
        clf.fit(
            trn_x, trn_y,
            eval_set=[(val_x, val_y)],
            eval_metric=lambda y_true,y_pred: tpr_weight_funtion(y_true,y_pred),
            early_stopping_rounds=200,
            verbose=200
        )
        oof[val_idx] += clf.predict_proba(val_x)[:, 1]
        train_df['prob'] += clf.predict_proba(train_df[cols])[:, 1] / skf.n_splits / len(seeds)
        test_df['prob'] += clf.predict_proba(test_df[cols])[:, 1] / skf.n_splits / len(seeds)
        df_importance = pd.DataFrame({
        'column':cols,
        'feature_importance':clf.feature_importances_
    })
        df_importance_list.append(df_importance)
    cv_auc = roc_auc_score(train_df['label'], oof)
    val_aucs.append(cv_auc)
    print('\ncv_auc: ', cv_auc)
print(val_aucs, np.mean(val_aucs))
```
模型重要度：
```python
df_importance = pd.concat(df_importance_list)
df_importance = df_importance.groupby(['column'])['feature_importance'].agg('mean').sort_values(ascending=False).reset_index()
df_importance.head(10)
```
## lgb原生
```python
result = []
ff = []
folds = 5
skf = StratifiedKFold(shuffle=True,n_splits=folds,random_state=1024)
is_test = 0
oof = np.zeros((len(train_df), 1))

for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df['tag'].values)):
    print('-----------' + str(fold) + '---------------')
    trn_data = train_df[train_idx]
    val_data = train_df[val_idx]
    #    print(val_data.shape, trn_data.shape)
    trn_label = train_df.loc[train_idx]['tag'].values
    val_label = train_df.loc[val_idx]['tag'].values

    params = {
        'learning_rate': 0.05,
        # 'boosting_type': 'dart',
        'objective': 'binary',
        #            'metric': 'binary_logloss',
        #            'metric': 'auc',
        'num_leaves': 31,
        'feature_fraction': 0.95,
        'bagging_fraction': 0.95,
        'bagging_freq': 5,
        # 'is_unbalance': True,
        'seed': 1,
        'bagging_seed': 1,
        'feature_fraction_seed': 7,
        'min_data_in_leaf': 5,
        'nthread': -1
    }

    dtrain = lgb.Dataset(trn_data, label=trn_label)
    dvalid = lgb.Dataset(val_data, label=val_label)
    save_path = os.path.join('model/model_' + model_name + str(fold) + '.pickle')
    if is_test == 0:
        clf = lgb.train(
            params=params,
            train_set=dtrain,
            num_boost_round=10000,
            valid_sets=[dvalid],
            early_stopping_rounds=100,
            verbose_eval=300,
            feval=self_metric
        )
        clf.fit(trn_data, trn_label, eval_set=(val_data, val_label), verbose=None)
    else:
        with open(save_path, 'rb+') as f:
            clf = pickle.load(f)
    oof[val_idx, 0] = clf.predict(val_data, num_iteration=clf.best_iteration)
    val_f1 = metrics.f1_score(val_label, list(map(lambda x: 1 if x > 0.5 else 0, oof[val_idx, 0])))
    print('Fold{} Best f1: {:.3f}'.format(fold + 1, val_f1))

    if is_test == 0:
        with open(save_path, 'wb') as f:
            pickle.dump(clf, f)

    ff.append(val_f1)

kfold_best_f1 = np.mean(ff)
print(kfold_best_f1)
pred_fold = np.zeros((len(test_df), 1))
for i in range(folds):
    save_path = 'model/model_' + model_name + str(i) + '.pickle'
    with open(save_path, 'rb+') as f:
        clf = pickle.load(f)
    pred_fold[:, 0] += clf.predict(test_df, num_iteration=clf.best_iteration) / folds
sub.to_csv('data/sub/sub_20211118_%.5f.csv' % kfold_best_f1, index=False)
```
## catboost 原生
```python
result = []
ff = []
mask_value = -9999
folds = 5
skf = StratifiedKFold(shuffle=True,n_splits=folds,random_state=1024)
is_test = 0
oof = np.zeros((len(train_df), 1))
for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df['tag'].values)):
    print('-----------' + str(fold) + '---------------')
    trn_data = train_df[train_idx]
    val_data = train_df[val_idx]
    trn_label = train_df.loc[train_idx]['tag'].values
    val_label = train_df.loc[val_idx]['tag'].values

    save_path = os.path.join('model/model_' + model_name + str(fold) + '.pickle')
    if is_test == 0:
        clf = cat.CatBoostClassifier(
                                     eval_metric='F1', use_best_model=True,
                                     early_stopping_rounds=500, random_state=2021,
                                     boosting_type='Plain', logging_level='Silent')
#                trn_data[np.isnan(trn_data)] = mask_value
        clf.fit(trn_data, trn_label, eval_set=(val_data, val_label), verbose=None)
    else:
        clf = cat.CatBoostClassifier(task_type='GPU', devices='0')
        clf.load_model(save_path)

    oof[val_idx, 0] = clf.predict(val_data)
    val_f1 = metrics.f1_score(val_label, list(map(lambda x: 1 if x > 0.5 else 0, oof[val_idx, 0])))
    print('Fold{} Best f1: {:.3f}'.format(fold+1,val_f1))
    if is_test == 0:
        with open(save_path, 'wb') as f:
            pickle.dump(clf, f)
    ff.append(val_f1)

pred_fold = np.zeros((len(test_df), 1))
for i in range(folds):
    save_path = 'model/model_' + model_name + str(i) + '.pickle'
    with open(save_path, 'rb+') as f:
        clf = pickle.load(f)
    pred_fold[:,0] += clf.predict_proba(test_df)[:,1]/folds
kfold_best_f1 = np.mean(ff)
sub.to_csv('data/sub/sub_20211118_%.5f.csv'%kfold_best_f1, index=False)
print(kfold_best_f1)
```
