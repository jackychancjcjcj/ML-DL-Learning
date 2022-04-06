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
## lgb
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
    num_leaves=100,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=1023,
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
model_name = 'lgb'
mask_value = -9999
df = pd.read_csv('data/trainreference.csv')
data = pd.read_csv('data/ml_feature.csv')
col = [str(i) for i in range(2004)]
data = data[col].values
folds = 5
skf = StratifiedKFold(n_splits=folds)
is_test = 0
oof = np.zeros((len(df), 1))
for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['tag'].values)):
    print('-----------' + str(fold) + '---------------')
    trn_data = data[train_idx]
    val_data = data[val_idx]
#    print(val_data.shape, trn_data.shape)
    trn_label = df.loc[train_idx]['tag'].values
    val_label = df.loc[val_idx]['tag'].values
    
    for j in ['tag']:
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
            if model_name == 'lgb':
                clf = lgb.train(
                    params=params,
                    train_set=dtrain,
                    num_boost_round=10000,
                    valid_sets=[dvalid],
                    early_stopping_rounds=100,
                    verbose_eval=300,
                    feval = self_metric
                )
                clf.fit(trn_data, trn_label, eval_set=(val_data, val_label), verbose=None)
        else:
            with open(save_path, 'rb+') as f:
                clf = pickle.load(f)
        if model_name == 'lgb':
            oof[val_idx, 0] = clf.predict(val_data, num_iteration=clf.best_iteration)
        val_f1 = metrics.f1_score(val_label, list(map(lambda x: 1 if x > 0.5 else 0, oof[val_idx, 0])))
        print('Fold{} Best f1: {:.3f}'.format(fold+1,val_f1))

        if is_test == 0:
            with open(save_path, 'wb') as f:
                pickle.dump(clf, f)
        
    ff.append(val_f1)
kfold_best_f1 = np.mean(ff)
print(kfold_best_f1)
pred_fold = np.zeros((len(data[len(df):]), 1))
for i in range(folds):
    save_path = 'model/model_' + model_name + str(i) + '.pickle'
    with open(save_path, 'rb+') as f:
        clf = pickle.load(f)
    pred_fold[:,0] += clf.predict(data[len(df):],num_iteration=clf.best_iteration)/folds
sub = pd.read_csv('data/sub/answer.csv')
sub['tag'] = pred_fold
sub.to_csv('data/sub/sub_20211118_%.5f.csv'%kfold_best_f1, index=False)
```
## 
