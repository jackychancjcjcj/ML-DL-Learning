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
        return xlf
      
    elif name == 'llf':
        llf=lgb.LGBMClassifier(
                learning_rate=0.05,
        #         n_estimators=10230,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=1023,
        ) 
        return llf
    
    elif name == 'clf':
        clf=cab.CatBoostClassifier(
            learning_rate=0.05,
        #     n_estimators=10230,
            subsample=0.8,
            random_seed=1023,
            eval_metric='AUC'
        )
        return clf
     
    elif name == 'rf':
        rf = RandomForestClassifier(
            oob_score=True, 
        #     n_estimators=10230,
        )
        return rf
        
def stacking(model_list,skf,train_df,test_df):

    cols = [col for col in train_df.columns if col not in ['id', 'label','prob_lgb']]
    df_importance_list = []
    train_df['prob_xgb'] = 0
    test_df['prob_xgb'] = 0
    train_df['prob_cab'] = 0
    test_df['prob_cab'] = 0
    train_df['prob_rf'] = 0
    test_df['prob_rf'] = 0
    
    xlf,llf,clf,rf = model_list[0],model_list[1],model_list[2],model_list[3]

    for i, (trn_idx, val_idx) in enumerate(skf.split(train_df, train_df['label'])):
        print('--------------------- {} fold ---------------------'.format(i))
        trn_x, trn_y = train_df[cols].iloc[trn_idx].reset_index(drop=True), train_df['label'].values[trn_idx]
        val_x, val_y = train_df[cols].iloc[val_idx].reset_index(drop=True), train_df['label'].values[val_idx]
        print('--------------------- xlf_training ---------------------')
        xlf.fit(
                trn_x, trn_y,
                eval_set=[(val_x, val_y)],
                eval_metric='auc',
                early_stopping_rounds=200,
                verbose=200
            )
        train_df['prob_xgb'] += xlf.predict_proba(train_df[cols])[:, 1] / skf.n_splits
        test_df['prob_xgb'] += xlf.predict_proba(test_df[cols])[:, 1] / skf.n_splits
        print('--------------------- llf_training ---------------------')
        llf.fit(
                trn_x, trn_y,
                eval_set=[(val_x, val_y)],
                eval_metric=lambda y_true,y_pred: tpr_weight_funtion(y_true,y_pred),
                early_stopping_rounds=200,
                verbose=200)
        train_df['prob_lgb'] += llf.predict_proba(train_df[cols])[:, 1] / skf.n_splits
        test_df['prob_lgb'] += llf.predict_proba(test_df[cols])[:, 1] / skf.n_splits
        print('--------------------- clf_training ---------------------')
        clf.fit(            
                trn_x, trn_y,
                eval_set=[(val_x, val_y)],
                early_stopping_rounds=200,
                verbose=200)
        train_df['prob_cab'] += clf.predict_proba(train_df[cols])[:, 1] / skf.n_splits
        test_df['prob_cab'] += clf.predict_proba(test_df[cols])[:, 1] / skf.n_splits
        print('--------------------- rf_training ---------------------')
        rf.fit(trn_x, trn_y)
        train_df['prob_rf'] += rf.predict_proba(train_df[cols])[:, 1] / skf.n_splits
        test_df['prob_rf'] += rf.predict_proba(test_df[cols])[:, 1] / skf.n_splits
        df_importance = pd.DataFrame({
        'column':cols,
        'xlf_feature_importance':xlf.feature_importances_,
        'llf_feature_importance':llf.feature_importances_,
        'clf_feature_importance':clf.feature_importances_,
        'rf_feature_importance':rf.feature_importances_,

    })
        df_importance_list.append(df_importance)
    return train_df,test_df,df_importance
    
def stacking_base_model(base_model,skf,train_df,test_df):
    oof = np.zeros(train_df.shape[0])
    df_importance_list = []
    cols = [col for col in train_df.columns if col not in ['id', 'label']]
    train_df['prob_final'] = 0
    test_df['prob_final'] = 0

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
            train_df['prob_final'] += clf.predict_proba(train_df[cols])[:, 1] / skf.n_splits / len(seeds)
            test_df['prob_final'] += clf.predict_proba(test_df[cols])[:, 1] / skf.n_splits / len(seeds)
            df_importance = pd.DataFrame({
            'column':cols,
            'feature_importance':clf.feature_importances_
        })
            df_importance_list.append(df_importance)
        cv_auc = roc_auc_score(train_df['label'], oof)
        val_aucs.append(cv_auc)
        print('\ncv_auc: ', cv_auc)
    print(val_aucs, np.mean(val_aucs))
    return train_df,test_df,df_importance_list
```
  
