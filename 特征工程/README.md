# 特征构造
# 目录
* [分箱特征](#1)
* [基本聚合特征](#2)
* [一度基本交叉特征](#3)
* [二度基本交叉特征](#4)
* [组合特征](#5)
* [偏离值特征](#6)
## <span id='1'>分箱特征</span>
```python
# ===================== amount_feas 分箱特征 ===============
for fea in tqdm(amount_feas, desc="分箱特征"):
    # 通过除法映射到间隔均匀的分箱中，每个分箱的取值范围都是loanAmnt/1000
    df['{}_bin1'.format(fea)] = np.floor_divide(df[fea], 1000)
    ## 通过对数函数映射到指数宽度分箱
    df['{}_bin2'.format(fea)] = np.floor(np.log10(df[fea]))
```
## <span id='2'>基本聚合特征</span>
```python
# ===================== amount_feas 基本聚合特征 ===============
for f in tqdm(amount_feas, desc="amount_feas 基本聚合特征"):
    for cate in category_fea:
        if f != cate:
            df['{}_{}_medi'.format(cate, f)] = df.groupby(cate)[f].transform('median')
            df['{}_{}_mean'.format(cate, f)] = df.groupby(cate)[f].transform('mean')
            df['{}_{}_max'.format(cate, f)] = df.groupby(cate)[f].transform('max')
            df['{}_{}_min'.format(cate, f)] = df.groupby(cate)[f].transform('min')
            df['{}_{}_std'.format(cate, f)] = df.groupby(cate)[f].transform('std')
```
## <span id='3'>一度基本交叉特征</span>
```python
# =================== amount_feas 一度基本交叉特征  =============================
agg_feas1 = []
for f1 in tqdm(amount_feas, desc="amount_feas 基本交叉特征"):
    for f2 in amount_feas:
        if f1 != f2:
            df[f1 + '_add_' + f2] = df[f1] + df[f2]
            df[f1 + '_diff_' + f2] = df[f1] - df[f2]
            df[f1 + '_div_' + f2] = df[f1] / df[f2]
            df[f1 + '_multi_' + f2] = df[f1] * df[f2]
            agg_feats1.append(f1 + '_add_' + f2)
            agg_feats1.append(f1 + '_diff_' + f2)
            agg_feats1.append(f1 + '_div_' + f2)
            agg_feats1.append(f1 + '_multi_' + f2)
```
## <span id='4'>二度基本交叉特征</span>
小心内存溢出：
```python
# =================== amount_feas 二度基本交叉特征  =============================
agg_feats2 = []
for f1 in tqdm(agg_feas1, desc="amount_feas 二度基本交叉特征"):
    for f2 in amount_feas:
        df[f1 + '_add_' + f2] = df[f1] + df[f2]
        df[f1 + '_diff_' + f2] = df[f1] - df[f2]
        df[f1 + '_div_' + f2] = df[f1] / df[f2]
        df[f1 + '_multi_' + f2] = df[f1] * df[f2]
        agg_feats2.append(f1 + '_add_' + f2)
        agg_feats2.append(f1 + '_diff_' + f2)
        agg_feats2.append(f1 + '_div_' + f2)
        agg_feats2.append(f1 + '_multi_' + f2)
```
## <span id='5'>组合特征</span>
挑选cate_cols和num_cols：
```python
cate_cols = ['GRZHZT','DWSSHY','DWJJLX','ZHIYE', 'ZHICHEN','XUELI','ZHIWU','HYZK']
num_cols = ['GRJCJS', 'GRZHYE', 'GRZHSNJZYE', 'GRZHDNGJYE', 'GRYJCE', 'DWYJCE','DKFFE', 'DKYE', 'DKLL']

for f in tqdm(cate_cols):
    df[f] = df[f].map(dict(zip(df[f].unique(), range(df[f].nunique()))))
    df[f + '_count'] = df[f].map(df[f].value_counts())
    df = pd.concat([df,pd.get_dummies(df[f],prefix=f"{f}")],axis=1)

cate_cols_combine = [[cate_cols[i], cate_cols[j]] for i in range(len(cate_cols)) \
                     for j in range(i + 1, len(cate_cols))]    

for f1, f2 in tqdm(cate_cols_combine):
    df['{}_{}_count'.format(f1, f2)] = df.groupby([f1, f2])['id'].transform('count')
    df['{}_in_{}_prop'.format(f1, f2)] = df['{}_{}_count'.format(f1, f2)] / df[f2 + '_count']
    df['{}_in_{}_prop'.format(f2, f1)] = df['{}_{}_count'.format(f1, f2)] / df[f1 + '_count']
```
## <span id='6'>偏离值特征</span>
挑选cate_cols和num_cols：
```python
cate_cols = ['HYZK', 'ZHIYE', 'ZHICHEN', 'ZHIWU', 'XUELI', 'DWJJLX', 'DWSSHY', 'GRZHZT']
num_cols = ['GRYJCE', 'DKFFE', 'DKLL', 'DKYE', 'GRJCJS', 'GRZHSNJZYE', 'GRZHDNGJYE']                   
for group in tqdm(cate_cols):
    for feature in num_cols:
        tmp = df.groupby(group)[feature].agg([sum, min, max, np.mean]).reset_index()
        tmp = pd.merge(df, tmp, on=group, how='left')
        df['{}-mean_gb_{}'.format(feature, group)] = df[feature] - tmp['mean']
        df['{}-min_gb_{}'.format(feature, group)] = df[feature] - tmp['min']
        df['{}-max_gb_{}'.format(feature, group)] = df[feature] - tmp['max']
        df['{}/sum_gb_{}'.format(feature, group)] = df[feature] / tmp['sum']
```
