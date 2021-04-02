## 标准代码
lgb：
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
        oof[val_idx] = clf.predict_proba(val_x)[:, 1]
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
