# 祖传参数
catboost回归：
```python
params = {
    'iterations':50000, 
    'learning_rate':0.01,
    'depth':6, 
    'l2_leaf_reg':3,
    'subsample':0.8,
    'loss_function':'RMSE',
    'eval_metric':'MAE',
    'cat_features':cat_features,
    #'bagging_temperature' : 0.2,
    #'use_best_model':True,
    'logging_level':'Verbose',
    'od_type':"Iter",
    'early_stopping_rounds':300,
    'random_seed':2021
    }
```
catboost分类：
```python
params = {
    'iterations':50000, 
    'learning_rate':0.01,
    'depth':6, 
    'l2_leaf_reg':3,
    'subsample':0.8,
    'loss_function':'Logloss',
    'eval_metric':'AUC',
    'cat_features':cat_features,
    #'bagging_temperature' : 0.2,
    #'use_best_model':True,
    'logging_level':'Verbose',
    'od_type':"Iter",
    'early_stopping_rounds':300,
    'random_seed':2021
    }
```
# 标准化代码
```python
import xgboost as xgb
import lightgbm as lgb
import catboost as cab
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score 

def model_setting(name):
    if name == 'xlf':
        xlf=xgb.XGBClassifier(
                learning_rate=0.05,
        #         n_estimators=10230,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=1023,
                objective='binary:logistic'
        )
        return name, xlf
      
    elif name == 'llf':
        llf=lgb.LGBMClassifier(
                learning_rate=0.05,
        #         n_estimators=10230,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=1023,
        ) 
        return name, llf
    
    elif name == 'clf':
        clf=cab.CatBoostClassifier(
            learning_rate=0.05,
        #     n_estimators=10230,
            subsample=0.8,
            random_seed=1023,
            eval_metric='AUC'
        )
        return name, clf
     
    elif name == 'rf':
        rf = RandomForestClassifier(
            oob_score=True, 
        #     n_estimators=10230,
        )
        return name, rf
        
def stacking(model_name,model,skf,train_df,test_df):
    
    oof = np.zeros(train_df.shape[0])
    cols = [col for col in train_df.columns if col not in ['id', 'label']]
    df_importance_list = []
    test_df['prob_{}'.format(model_name)] = 0

    for i, (trn_idx, val_idx) in enumerate(skf.split(train_df, train_df['label'])):
        print('--------------------- {} fold ---------------------'.format(i))
        trn_x, trn_y = train_df[cols].iloc[trn_idx].reset_index(drop=True), train_df['label'].values[trn_idx]
        val_x, val_y = train_df[cols].iloc[val_idx].reset_index(drop=True), train_df['label'].values[val_idx]
   
        if model_name == 'xlf':
            print('--------------------- xlf_training ---------------------')
            model.fit(
                    trn_x, trn_y,
                    eval_set=[(val_x, val_y)],
                    eval_metric='auc',
                    early_stopping_rounds=200,
                    verbose=200
                )
            test_df['prob_{}'.format(model_name)] += model.predict_proba(test_df[cols])[:, 1] / skf.n_splits
            oof[val_idx] = model.predict_proba(val_x)[:, 1]
        elif model_name == 'llf':
            print('--------------------- llf_training ---------------------')
            model.fit(
                    trn_x, trn_y,
                    eval_set=[(val_x, val_y)],
                    eval_metric=lambda y_true,y_pred: tpr_weight_funtion(y_true,y_pred), # 自定义的评价函数
                    early_stopping_rounds=200,
                    verbose=200)
            test_df['prob_{}'.format(model_name)] += model.predict_proba(test_df[cols])[:, 1] / skf.n_splits
            oof[val_idx] = model.predict_proba(val_x)[:, 1]
        elif model_name == 'clf':
            print('--------------------- clf_training ---------------------')
            model.fit(            
                    trn_x, trn_y,
                    eval_set=[(val_x, val_y)],
                    early_stopping_rounds=200,
                    verbose=200)
            test_df['prob_{}'.format(model_name)] += model.predict_proba(test_df[cols])[:, 1] / skf.n_splits
            oof[val_idx] = model.predict_proba(val_x)[:, 1]
        elif model_name == 'rf':
            print('--------------------- rf_training ---------------------')
            model.fit(trn_x, trn_y)
            test_df['prob_{}'.format(model_name)] += model.predict_proba(test_df[cols])[:, 1] / skf.n_splits
            oof[val_idx] = model.predict_proba(val_x)[:, 1]
        df_importance = pd.DataFrame({
        'column':cols,
        '{}_feature_importance'.format(model_name):model.feature_importances_,
        })
    train_df['prob_{}'.format(model_name)] = oof
    return df_importance,oof
    
def stacking_base_model(base_model_name,base_model,skf,train_df,test_df):
    oof = np.zeros(train_df.shape[0])
    df_importance_list = []
    cols = [col for col in train_df.columns if col not in ['id', 'label']]
    test_df['prob_final'] = 0

    val_aucs = []
    seeds = [1023, 2048, 2098]
    for seed in seeds:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for i, (trn_idx, val_idx) in enumerate(skf.split(train_df, train_df['label'])):
            print('--------------------- {} fold ---------------------'.format(i))
            trn_x, trn_y = train_df[cols].iloc[trn_idx].reset_index(drop=True), train_df['label'].values[trn_idx]
            val_x, val_y = train_df[cols].iloc[val_idx].reset_index(drop=True), train_df['label'].values[val_idx]
            base_model.fit(
                trn_x, trn_y,
                eval_set=[(val_x, val_y)],
                eval_metric=lambda y_true,y_pred: tpr_weight_funtion(y_true,y_pred),
                early_stopping_rounds=200,
                verbose=200
            )
            oof[val_idx] += base_model.predict_proba(val_x)[:, 1] / len(seeds)
            test_df['prob_final'] += base_model.predict_proba(test_df[cols])[:, 1] / skf.n_splits / len(seeds)
            df_importance = pd.DataFrame({
            'column':cols,
            'feature_importance':base_model.feature_importances_
        })
            df_importance_list.append(df_importance)
        cv_auc = roc_auc_score(train_df['label'], oof)
        val_aucs.append(cv_auc)
        print('\ncv_auc: ', cv_auc)
    df_importance = pd.concat(df_importance_list)
    df_importance = df_importance.groupby(['column'])['feature_importance'].agg('mean').sort_values(ascending=False).reset_index()
    train_df['prob_final'] = oof
    print(val_aucs, np.mean(val_aucs))
    return df_importance,oof
```
  
