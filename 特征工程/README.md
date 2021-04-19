# 特征构造
特征构造顺序：  
1. 预处理  
2. 聚合特征（group）
3. 统计特征（一度，多度交叉，偏离值，频率）
4. 编码特征
# 目录
* [分箱特征](#1)
* [基本聚合特征](#2)
* [一度基本交叉特征](#3)
* [二度基本交叉特征](#4)
* [组合特征](#5)
* [偏离值特征](#6)
* [频率特征](#7)
* [目标编码](#8)
* [TF-IDF编码](#9)
* [W2V编码](#10)
* [经纬度特征1](https://mp.weixin.qq.com/s?__biz=Mzk0NDE5Nzg1Ng==&mid=2247490133&idx=1&sn=036127fcb121257ec9c57c47b55503bc&source=41#wechat_redirect)
* [经纬度特征2](https://mp.weixin.qq.com/s?__biz=Mzk0NDE5Nzg1Ng==&mid=2247490131&idx=1&sn=ecbff9ecf4692e7af97b30fe1f431e2f&source=41#wechat_redirect)
* [熵](#11)
## <span id='1'>分箱特征</span>
```python
# ===================== amount_feas 分箱特征 ===============
for fea in tqdm(amount_feas, desc="分箱特征"):
    # 通过除法映射到间隔均匀的分箱中，每个分箱的取值范围都是loanAmnt/1000
    df['{}_bin1'.format(fea)] = np.floor_divide(df[fea], 1000)
    ## 通过对数函数映射到指数宽度分箱
    df['{}_bin2'.format(fea)] = np.floor(np.log10(df[fea]))
    

# ===================== amount_feas 分箱特征 ===============
for fea in tqdm(['民宿评分'], desc="分箱特征"):
    ## qcut分箱，按数据量分箱
    qcut_labels = [1,2,3,4,5]
    df['{}_qcut'.format(fea)] = pd.qcut(df[fea].rank(method='first'),q=[0,.2,.4,.6,.8,1],labels=qcut_labels)
    ## cut分箱，按数字边界分箱
    cut_labels = [1,2,3,4]
    cut_bins = [0,60,80,90,100]
    df['{}_cut'.format(fea)] = pd.cut(df[fea],bins=cut_bins,labels=cut_labels)
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
            df['{}_{}_var'.format(cate, f)] = df.groupby(cate)[f].transform('var')
            df['{}_{}_mode'.format(cate, f)] = df.groupby(cate)[f].transform('mode')
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
## <span id='7'>频率特征</span>
```python
for f in cols:
    vc = df[f].value_counts(dropna=True,normalize=True).to_dict()
    df[f'{f}_freq'] = df[f].map(vc)
```
## <span id='8'>目标编码</span>
常规target encoding：
```python
import gc
def stat(df, df_merge, group_by, agg):
    group = df.groupby(group_by).agg(agg)

    columns = []
    for on, methods in agg.items():
        for method in methods:
            columns.append('{}_{}_{}'.format('_'.join(group_by), on, method))
    group.columns = columns
    group.reset_index(inplace=True)
    df_merge = df_merge.merge(group, on=group_by, how='left')

    del (group)
    gc.collect()
    return df_merge


def statis_feat(df_know, df_unknow):
    df_unknow = stat(df_know, df_unknow, ['民宿评分'], {'价格': ['mean', 'std', 'max']})
    df_unknow = stat(df_know, df_unknow, ['邮编'], {'价格': ['mean', 'std', 'max']})

    return df_unknow


# 5折交叉
df_train = df_features[~df_features['价格'].isnull()].reset_index(drop=True)
df_test = df_features[df_features['价格'].isnull()].reset_index(drop=True)

df_stas_feat = None
kf = KFold(n_splits=5, random_state=2021, shuffle=True)
for train_index, val_index in kf.split(df_train):
    df_fold_train = df_train.iloc[train_index]
    df_fold_val = df_train.iloc[val_index]

    df_fold_val = statis_feat(df_fold_train, df_fold_val)
    df_stas_feat = pd.concat([df_stas_feat, df_fold_val], axis=0)

    del(df_fold_train)
    del(df_fold_val)
    gc.collect()

df_test = statis_feat(df_train, df_test)
df_features = pd.concat([df_stas_feat, df_test], axis=0)

del(df_stas_feat)
del(df_train)
del(df_test)
gc.collect()
```
beta target encoding：  
[资料](https://mp.weixin.qq.com/s?__biz=Mzk0NDE5Nzg1Ng==&mid=2247494163&idx=1&sn=5aba9ce08911f12b06619226809cdc7d&chksm=c32af39cf45d7a8aad50b7dfc58dc6eee6128d7f9c0998b86cdfb45f1419427b5ca6a4e0a3ec&mpshare=1&scene=1&srcid=0412roAaBBpbfWHKcbOgezod&sharer_sharetime=1618190773139&sharer_shareid=9b869c9a24181fe91d7ddd3f39c6511b&version=3.1.6.3605&platform=win#rd)  
```python
import numpy as np 
import pandas as pd
from sklearn.preprocessing import LabelEncoder

'''
    代码摘自原作者：https://www.kaggle.com/mmotoki/beta-target-encoding
'''
class BetaEncoder(object):
        
    def __init__(self, group):
        
        self.group = group
        self.stats = None
        
    # get counts from df
    def fit(self, df, target_col):
        # 先验均值
        self.prior_mean = np.mean(df[target_col]) 
        stats           = df[[target_col, self.group]].groupby(self.group)
        # count和sum
        stats           = stats.agg(['sum', 'count'])[target_col]    
        stats.rename(columns={'sum': 'n', 'count': 'N'}, inplace=True)
        stats.reset_index(level=0, inplace=True)           
        self.stats      = stats
        
    # extract posterior statistics
    def transform(self, df, stat_type, N_min=1):
        
        df_stats = pd.merge(df[[self.group]], self.stats, how='left')
        n        = df_stats['n'].copy()
        N        = df_stats['N'].copy()
        
        # fill in missing
        nan_indexs    = np.isnan(n)
        n[nan_indexs] = self.prior_mean
        N[nan_indexs] = 1.0
        
        # prior parameters
        N_prior     = np.maximum(N_min-N, 0)
        alpha_prior = self.prior_mean*N_prior
        beta_prior  = (1-self.prior_mean)*N_prior
        
        # posterior parameters
        alpha       =  alpha_prior + n
        beta        =  beta_prior  + N-n
        
        # calculate statistics
        if stat_type=='mean':
            num = alpha
            dem = alpha+beta
                    
        elif stat_type=='mode':
            num = alpha-1
            dem = alpha+beta-2
            
        elif stat_type=='median':
            num = alpha-1/3
            dem = alpha+beta-2/3
        
        elif stat_type=='var':
            num = alpha*beta
            dem = (alpha+beta)**2*(alpha+beta+1)
                    
        elif stat_type=='skewness':
            num = 2*(beta-alpha)*np.sqrt(alpha+beta+1)
            dem = (alpha+beta+2)*np.sqrt(alpha*beta)

        elif stat_type=='kurtosis':
            num = 6*(alpha-beta)**2*(alpha+beta+1) - alpha*beta*(alpha+beta+2)
            dem = alpha*beta*(alpha+beta+2)*(alpha+beta+3)
            
        # replace missing
        value = num/dem
        value[np.isnan(value)] = np.nanmedian(value)
        return value
```
```python
N_min = 1000
feature_cols = []    

# encode variables
for c in cat_cols:

    # fit encoder
    be = BetaEncoder(c)
    be.fit(train, 'deal_probability')

    # mean
    feature_name = f'{c}_mean'
    train[feature_name] = be.transform(train, 'mean', N_min)
    test[feature_name]  = be.transform(test,  'mean', N_min)
    feature_cols.append(feature_name)

    # mode
    feature_name = f'{c}_mode'
    train[feature_name] = be.transform(train, 'mode', N_min)
    test[feature_name]  = be.transform(test,  'mode', N_min)
    feature_cols.append(feature_name)
    
    # median
    feature_name = f'{c}_median'
    train[feature_name] = be.transform(train, 'median', N_min)
    test[feature_name]  = be.transform(test,  'median', N_min)
    feature_cols.append(feature_name)    

    # var
    feature_name = f'{c}_var'
    train[feature_name] = be.transform(train, 'var', N_min)
    test[feature_name]  = be.transform(test,  'var', N_min)
    feature_cols.append(feature_name)        
    
    # skewness
    feature_name = f'{c}_skewness'
    train[feature_name] = be.transform(train, 'skewness', N_min)
    test[feature_name]  = be.transform(test,  'skewness', N_min)
    feature_cols.append(feature_name)    
    
    # kurtosis
    feature_name = f'{c}_kurtosis'
    train[feature_name] = be.transform(train, 'kurtosis', N_min)
    test[feature_name]  = be.transform(test,  'kurtosis', N_min)
    feature_cols.append(feature_name)  
```

## <span id='9'>TF-IDF编码</span>
```python
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
df_features['便利设施'] = df_features['便利设施'].apply(
    lambda x: x.replace('{', '').replace('}', '').replace('"', '').replace(':', '').replace(',', ' '))
# df_features['便利设施'] = df_features['便利设施'].str.lower()

n_components = 12

X = list(df_features['便利设施'].values)
tfv = TfidfVectorizer(ngram_range=(1,1), max_features=10000)
tfv.fit(X)
X_tfidf = tfv.transform(X)
svd = TruncatedSVD(n_components= n_components)
svd.fit(X_tfidf)
X_svd = svd.transform(X_tfidf)

for i in range(n_components):
    df_features[f'便利设施_tfidf_{i}'] = X_svd[:, i]
```
## <span id='10'>W2V编码</span>
```python
from gensim.models import word2vec
emb_size = 4
sentences = df_features['便利设施'].values.tolist()

words = []
for i in range(len(sentences)):
    sentences[i] = sentences[i].split()
    words += sentences[i]

words = list(set(words))

model = word2vec.Word2Vec(sentences, size=emb_size, window=3,
                 min_count=1, sg=0, hs=1, seed=2021)

emb_matrix_mean = []
emb_matrix_max = []

for seq in sentences:
    vec = []
    for w in seq:
        if w in model:
            vec.append(model[w])
    if len(vec) > 0:
        emb_matrix_mean.append(np.mean(vec, axis=0))
        emb_matrix_max.append(np.max(vec, axis=0))
    else:
        emb_matrix_mean.append([0] * emb_size)
        emb_matrix_max.append([0] * emb_size)

df_emb_mean = pd.DataFrame(emb_matrix_mean)
df_emb_mean.columns = ['便利设施_w2v_mean_{}'.format(
    i) for i in range(emb_size)]

df_emb_max = pd.DataFrame(emb_matrix_max)
df_emb_max.columns = ['便利设施_w2v_max_{}'.format(
    i) for i in range(emb_size)]

for i in range(emb_size):
    df_features[f'便利设施_w2v_mean_{i}'] = df_emb_mean[f'便利设施_w2v_mean_{i}']
    df_features[f'便利设施_w2v_max_{i}'] = df_emb_max[f'便利设施_w2v_max_{i}']

df_features.head()
```
## <span id='11'>熵</span>
```python
from scipy.stats import entropy
df = df.merge(df.groupby(cate_feature, as_index=False)[value].agg({
            '{}_{}_nunique'.format(cate_feature, value): 'nunique',
            '{}_{}_ent'.format(cate_feature, value): lambda x: entropy(x.value_counts() / x.shape[0])
        }), on=cate_feature, how='left')
```
