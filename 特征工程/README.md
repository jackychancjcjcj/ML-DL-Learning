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
* [CountVec编码](#18)
* [经纬度特征1](https://mp.weixin.qq.com/s?__biz=Mzk0NDE5Nzg1Ng==&mid=2247490133&idx=1&sn=036127fcb121257ec9c57c47b55503bc&source=41#wechat_redirect)
* [经纬度特征2](https://mp.weixin.qq.com/s?__biz=Mzk0NDE5Nzg1Ng==&mid=2247490131&idx=1&sn=ecbff9ecf4692e7af97b30fe1f431e2f&source=41#wechat_redirect)
* [熵+nunique值](#11)
* [对匿名特征暴力统计特征](#12)
* [woe编码](#13)
* [加窗口的聚合特征](#14)
* [普通统计特征](#15)
* [catboost类别编码](#16)
* [条件特征](#17)
* [缺失值组合特征](#19)
* [手动构造行为序列+w2v编码](#20)
* [黄金组合特征](#21)
* [数据倾斜](#22)
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
    
    
# ===================== amount_feas 卡方分箱（有监督） ===============
def chi2(arr):
    '''
    计算卡方值
    arr：频数统计表，二维numpy数组
    '''
    assert(arr.ndim == 2) 
    # assert断言是声明其布尔值必须为真的判定，如果发生异常就说明表达示为假。
    # 可以理解assert断言语句为raise-if-not，用来测试表示式，其返回值为假，就会触发异常。
    
    R_N = arr.sum(axis=1) # 计算每行总频数
    C_N = arr.sum(axis=0) # 计算每列总频数
    N = arr.sum() # 总频数
    
    E = np.ones(arr.shape)*C_N/N # 计算期望频数
    E = (E.T*R_N).T
    square = (arr-E)**2/E
    square[E==0] = 0 # 当期望频数为0时，作为分母无意义，不计入卡方值
    
    v = square.sum() # 卡方值
    return v
def chiMerge(df,col,target,max_groups = None,threshold = None):
    '''
    卡方分箱
    df: pandas DataFrame 数据集
    col: 需要分箱的变量名（数值型）
    target：类标签
    max_groups: 最大分组数
    threshold： 卡方阈值，如果未指定max_groups，默认使用置信度95%设置threshold
    return：包括各组的起始值的列表
    '''
    freq_tab = pd.crosstab(df[col],df[target])
    
    # 转成numpy数组用于计算
    freq = freq_tab.values
    
    # 初始分组切分点，每个变量值都是切分点，每组中只包含一个变量值
    # 分组区间是左闭右开的，如cutoffs = [1,2,3]，表示区间，[1,2),[2,3).[3,3+)
    cutoffs = freq_tab.index.values
    
    # 如果没有指定最大分组
    if max_groups is None:
        # 如果没有指定卡方阈值，就以95%的置信度（自由度为类数目-1）设定阈值
        if threshold is None:
            # 类数目
            cls_num = freq.shape[-1]
            threshold = chi2.isf(0.05,df = cls_num-1)
    
    while True:
        minvalue = None
        minidx = None
        # 从第一组开始，依次计算两组卡方值，并判断是否小于当前最小的卡方
        for i in range(len(freq)-1):
            v = chi2(freq[i:i+2])
            if minvalue is None or minvalue > v: # 小于当前最小卡方，更换最小值
                minvalue = v
                minidx = i
        
        # 如果最小卡方值小于阈值，则合并最小卡方值的相邻两组，并继续循环
        if (max_groups is not None and max_groups < len(freq)) or (threshold is not None and minvalue < threshold):
            # minidx 后一行合并到minidx
            tmp = freq[minidx] + freq[minidx+1]
            freq[minidx] = tmp
            # 删除minidx后一行
            freq = np.delete(freq,minidx+1,0)
            # 删除对应的切分点
            cutoffs = np.delete(cutoffs,minidx+1,0)
            
        else: # 最小卡方值不小于阈值，停止合并
            break
    return cutoffs
def value2group(x,cutoffs):
    '''
    将变量的值转换成相应的组
    x: 需要转换到分组的值
    cutoffs：各组的起始值
    return： x对应的组，如group1，从group1开始

    '''
    # 切分点从小到大排序
    cutoffs = sorted(cutoffs)
    num_groups = len(cutoffs)
    # 异常情况：小于第一组的起始值，这里直接放到第一组
    # 异常值建议在分组之前先处理妥善
    if x < cutoffs[0]:
        return 'group1'
    
    for i in range(1,num_groups):
        if cutoffs[i-1] <= x <cutoffs[i]:
            return 'group{}'.format(i)
    # 最后一组，也可能会包括一些非常大的异常值
    return 'group{}'.format(num_groups)
cutoffs = chiMerge(df[df['y1_is_purchase'].notnull()],'newvalue','y1_is_purchase',max_groups=8)
df['total_newvalue_group']=df['newvalue'].apply(value2group,args=(cutoffs,))
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
            df['{}_{}_sum'.format(cate, f)] = df.groupby(cate)[f].transform('sum')
            df['{}_{}_skew'.format(cate, f)] = df.groupby(cate)[f].transform('skew')
            df['{}_{}_rank'.format(cate, f)] = df.groupby(cate)[f].transform('rank')
            df['{}_{}_nunique'.format(cate, f)] = df.groupby(cate)[f].transform('nunique')
            df['{}_{}_q1'.format(cate, f)] = df.groupby(cate)[f].transform(lambda x: x.quantile(0.25))
            df['{}_{}_q3'.format(cate, f)] = df.groupby(cate)[f].transform(lambda x: x.quantile(0.75))
            df['{}_{}_qsub'.format(cate, f)] = df.groupby(cate)[f].transform(lambda x: x.quantile(0.75) - x.quantile(0.25))
            
for f in tqdm(int_feas, desc="amount_feas 基本聚合特征"):
    for cate in category_fea:
        if f != cate:
            df['{}_{}_count'.format(cate, f)] = df.groupby(cate)[f].transform('count')
            df['{}_{}_nunique'.format(cate, f)] = df.groupby(cate)[f].transform('nunique')
            df['{}_{}_nunique_count'.format(cate, f)] = df['{}_{}_nunique'.format(cate, f)] / df['{}_{}_count'.format(cate, f)]
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
        df[f1 + '_div_' + f2] = df[f1] / (df[f2]+1e-3)
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

#类别特征组合编码
for f1,f2 in [[]]:
    df['{}_{}'.format(f1,f2)] = df[f1].map(str) + '_' + df[f2].map(str)

for f in tqdm(cate_cols):
    df[f+'_nunique'] = df[f].map(dict(zip(df[f].unique(), range(df[f].nunique()))))
    df[f+'_count'] = df[f].map(df[f].value_counts())
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
## <span id='11'>熵+nunique值</span>
```python
from scipy.stats import entropy
df = df.merge(df.groupby(cate_feature, as_index=False)[value].agg({
            '{}_{}_nunique'.format(cate_feature, value): 'nunique',
            '{}_{}_ent'.format(cate_feature, value): lambda x: entropy(x.value_counts() / x.shape[0])
        }), on=cate_feature, how='left')
```
## <span id='12'>对匿名特征暴力统计特征</span>
```python
#求熵
def myEntro(x):
    """
        calculate shanno ent of x
    """
    x = np.array(x)
    x_value_list = set([x[i] for i in range(x.shape[0])])
    ent = 0.0
    for x_value in x_value_list:
        p = float(x[x == x_value].shape[0]) / x.shape[0]
        logp = np.log2(p)
        ent -= p * logp
    #     print(x_value,p,logp)
    # print(ent)
    return ent

#求均方根
def myRms(records):
    records = list(records)
    """
    均方根值 反映的是有效值而不是平均值
    """
    return np.math.sqrt(sum([x ** 2 for x in records]) / len(records))

#求取众数
def myMode(x):
    return np.mean(pd.Series.mode(x))
    
#分别求取10，25，75，90分位值
def myQ25(x):
    return x.quantile(0.25)
    
def myQ75(x):
    return x.quantile(0.75)

def myQ10(x):
    return x.quantile(0.25)
    
def myQ90(x):
    return x.quantile(0.75)
    
#求值的范围
def myRange(x):
    return pd.Series.max(x) - pd.Series.min(x)

n_feat = ['n0', 'n1', 'n2', 'n4', 'n5', 'n6', 'n7', 'n8', 'n9', 'n10', 'n11', 'n12', 'n13', 'n14']
nameList = ['min', 'max', 'sum', 'mean', 'median', 'skew', 'std', 'mode', 'range', 'Q25','Q75']
statList = ['min', 'max', 'sum', 'mean', 'median', 'skew', 'std', myMode, myRange, myQ25, myQ75]

for i in tqdm(range(len(nameList))):
    df['n_feat_{}'.format(nameList[i])] = df[n_feat].agg(statList[i], axis=1)
print('n特征处理后：', df.shape)
```
## <span id='13'>woe编码</span>
```python
def CalWOE(df, fea, label):
    eps = 0.000001
    gbi = pd.crosstab(df[fea], df[label]) + eps
    gb = df[label].value_counts() + eps
    gbri = gbi / gb
    gbri['woe'] = np.log(gbri[1] / gbri[0])

    return gbri['woe']

#woe编码
def woe_feature(train, test, feats, k):
    folds = StratifiedKFold(n_splits=k, shuffle=True, random_state=2020)  # 这里最好和后面模型的K折交叉验证保持一致

    train['fold'] = None
    for fold_, (trn_idx, val_idx) in enumerate(folds.split(train, train['label'])):
        train.loc[val_idx, 'fold'] = fold_

    kfold_features = []
    for feat in feats:
        nums_columns = ['label']
        for f in nums_columns:
            colname = feat + '_' + f + '_woe'
            kfold_features.append(colname)
            train[colname] = None
            for fold_, (trn_idx, val_idx) in enumerate(folds.split(train, train['label'])):
                tmp_trn = train.iloc[trn_idx]
                #order_label = tmp_trn.groupby([feat])[f].mean()
                order_label = CalWOE(tmp_trn, feat, f)
                tmp = train.loc[train.fold == fold_, [feat]]
                train.loc[train.fold == fold_, colname] = tmp[feat].map(order_label)
                # fillna
                global_mean = order_label.mean()
                train.loc[train.fold == fold_, colname] = train.loc[train.fold == fold_, colname].fillna(global_mean)
            train[colname] = train[colname].astype(float)

        for f in nums_columns:
            colname = feat + '_' + f + '_woe'
            test[colname] = None
            #order_label = train.groupby([feat])[f].mean()
            order_label = CalWOE(train, feat, f)
            test[colname] = test[feat].map(order_label)
            # fillna
            global_mean = order_label.mean()
            test[colname] = test[colname].fillna(global_mean)
            test[colname] = test[colname].astype(float)
    del train['fold']
    return train, test
```
## <span id='14'>加窗口的聚合特征</span>
```python
group_df = df[df['days_diff']>window].groupby('user')['amount'].agg({
    'user_amount_mean_{}d'.format(window): 'mean',
    'user_amount_std_{}d'.format(window): 'std',
    'user_amount_max_{}d'.format(window): 'max',
    'user_amount_min_{}d'.format(window): 'min',
    'user_amount_sum_{}d'.format(window): 'sum',
    'user_amount_med_{}d'.format(window): 'median',
    'user_amount_cnt_{}d'.format(window): 'count',
    # 'user_amount_q1_{}d'.format(window): lambda x: x.quantile(0.25),
    # 'user_amount_q3_{}d'.format(window): lambda x: x.quantile(0.75),
    # 'user_amount_qsub_{}d'.format(window): lambda x: x.quantile(0.75) - x.quantile(0.25),
    # 'user_amount_skew_{}d'.format(window): 'skew',
    # 'user_amount_q4_{}d'.format(window): lambda x: x.quantile(0.8),
    # 'user_amount_q5_{}d'.format(window): lambda x: x.quantile(0.3),
    # 'user_amount_q6_{}d'.format(window): lambda x: x.quantile(0.7),
    }).reset_index()
df = df.merge(group_df, on=['user'], how='left')
```
## <span id='15'>普通统计特征</span>
```python
df['nan_num'] = df.isnull().sum(axis=1)

tmp = [['人均床数量','人均卧室量'],['卧室床均量','人均卧室量']]
for fea in tmp:
    df[f'{fea[0]}_{fea[1]}_std'] = df[fea].std(1)
    df[f'{fea[0]}_{fea[1]}_max'] = df[fea].max(1)
    df[f'{fea[0]}_{fea[1]}_min'] = df[fea].min(1)
    df[f'{fea[0]}_{fea[1]}_sub'] = df[fea[0]] - df[fea[1]]
del tmp
gc.collect()

#计算空值
df['is_null'] = 0
df.loc[df[value].isnull(), 'is_null'] = 1
group_df = df.groupby(['user'])['is_null'].agg({'user_{}_{}_null_cnt'.format(prefix, value): 'sum',
                                                'user_{}_{}_null_ratio'.format(prefix, value): 'mean'}).reset_index()

```
## <span id='16'>catboost类别编码</span>
```python
def cat_encoding(train, test, k ,feature):
    #feature = [f for f in train.select_dtypes('object').columns if f not in ['user']]
    folds = StratifiedKFold(n_splits=k, shuffle=True, random_state=2020)  # 这里最好和后面模型的K折交叉验证保持一致

    train['fold'] = None
    for fold_, (trn_idx, val_idx) in enumerate(folds.split(train, train['label'])):
        train.loc[val_idx, 'fold'] = fold_

    enc = CatEncode.CatBoostEncoder()

    for feat in feature:
        nums_columns = ['label']
        for f in nums_columns:
            colname = feat + '_' + f + '_cat_enc'

            train[colname] = None
            for fold_, (trn_idx, val_idx) in enumerate(folds.split(train, train['label'])):
                tmp_trn = train.iloc[trn_idx]
                #order_label = tmp_trn.groupby([feat])[f].mean()
                order_label = enc.fit_transform(tmp_trn[feat], tmp_trn['label']).values.squeeze()
                enc_dic = dict(zip(tmp_trn[feat], order_label))
                tmp = train.loc[train.fold == fold_, [feat]]
                train.loc[train.fold == fold_, colname] = tmp[feat].map(enc_dic)
                # fillna
                global_mean = order_label.mean()
                train.loc[train.fold == fold_, colname] = train.loc[train.fold == fold_, colname].fillna(global_mean)
            train[colname] = train[colname].astype(float)

        for f in nums_columns:
            colname = feat + '_' + f + '_cat_enc'
            test[colname] = None
            order_label = enc.fit_transform(train[feat], train['label']).values.squeeze()
            enc_dic = dict(zip(train[feat], order_label))
            test[colname] = test[feat].map(enc_dic)
            # fillna
            global_mean = order_label.mean()
            test[colname] = test[colname].fillna(global_mean)
            test[colname] = test[colname].astype(float)
        #del train[feat]
    del train['fold']
    return train, test
```
## <span id='17'>条件特征</span>
```python
for wd in range(7):
    data_tmp = trans[trans['week'] == wd].groupby('user')['days_diff'].count().reset_index()
    data_tmp = pd.DataFrame(data_tmp)
    data_tmp.columns = ['user', 'trans_user_week_{}_cnt'.format(wd)]
    df = df.merge(data_tmp, on=['user'], how='left')

time_period = [-1, 8, 12, 15, 23]
for tp in range(4):
    data_tmp = pd.DataFrame(trans[((trans['hour'] > time_period[tp]) & (trans['hour'] < time_period[tp + 1]))]. \
                            groupby('user')['days_diff'].count().reset_index())
    data_tmp.columns = ['user', 'trans_user_time_period_{}_cnt'.format(tp)]
    df = df.merge(data_tmp, on=['user'], how='left')
        
for col in tqdm(['op_type', 'op_mode', 'net_type', 'channel']):
    df_temp = df_op[['user', 'hour', col]].copy()
    df_temp = df_temp.pivot_table(index='user', columns=col,
                                  values='hour', aggfunc=['mean', 'std', 'max', 'min']).fillna(0)
    df_temp.columns = ['op_{}_{}_hour_{}'.format(col, f[1], f[0]) for f in df_temp.columns]
    df_temp.reset_index(inplace=True)
    df_temp.rename({'index': 'user'}, inplace=True, axis=1)
    df_feature = df_feature.merge(df_temp, how='left')
```
## <span id='18'>CountVec编码</span>
```python
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer

df_features['便利设施'] = df_features['便利设施'].apply(
    lambda x: x.replace('{', '').replace('}', '').replace('"', '').replace(':', '').replace(',', ' '))
# df_features['便利设施'] = df_features['便利设施'].str.lower()
df_features['便利设施'] = df_features['便利设施'].apply(lambda x: ','.join(x))
n_components = 12

env = CountVectorizer()
X_tfidf = env.fit_transform(df_features['便利设施'])
svd = TruncatedSVD(n_components= n_components)
X_svd = svd.fit_transform(X_tfidf)

for i in range(n_components):
    df_features[f'便利设施_countvec_{i}'] = X_svd[:, i]
```
## <span id='19'>缺失值组合特征</span>
```python
#4， 缺失值统计，统计存在缺失值的特征，构造缺失值相关计数特征
loss_fea = ['bankCard','residentAddr','highestEdu','linkRela']
for i in loss_fea:
    a = data.loc[data[i]==-999]
    e = a.groupby(['certId'])['id'].count().reset_index(name=i+'_certId_count') 
    data = data.merge(e,on='certId',how='left')
    
    d = a.groupby(['loanProduct'])['id'].count().reset_index(name=i+'_loan_count') 
    data = data.merge(d,on='loanProduct',how='left')
    
    m = a.groupby(['job'])['id'].count().reset_index(name=i+'_job_count') 
    data = data.merge(m,on='job',how='left')
    
    data['certloss_'+i] = data[i+'_certId_count']/data['certId_count']
    data['jobloss_'+i] = data[i+'_job_count']/data['job_count']
```
## <span id='20'>手动构造行为序列+w2v编码</span>
```python
from gensim.models import Word2Vec
def w2v_transform(X,word2vec,length):
    length = len(base_col[3:])
    return np.array([np.hstack([
            np.mean([word2vec[w] 
                     for w in words if w in word2vec] or
                    [np.zeros(length)], axis=1)
        ,   np.max([word2vec[w] 
                     for w in words if w in word2vec] or
                    [np.zeros(length)], axis=1)
                ])   for words in X
        
        ])

def get_w2v(data_frame,feat,length):
    model = Word2Vec(data_frame[feat].values, size=length, window=20, min_count=1,
                     workers=10, iter=10)
    return model
    
def w2v_feat(data):
    tr_w2v = get_w2v(data[['rid']],'rid',50)
    vect = w2v_transform(data.rid.values,tr_w2v.wv,50)
    for i in range(vect.shape[1]):
        data['w2vn'+str(i)] = vect[:,i]
    return data

zx_col = ['x_'+str(i) for i in range(78)] # 需要构建的特征
tmp = df[zx_col].corr()
drop_col = []
base_col = []
for i in zx_col:
    base_col.append(i)
    tmp1 = tmp[i]
    tmp2 = tmp1[tmp1==1].index.tolist()
    tmp2 = [n for n in tmp2 if n not in base_col]
    drop_col.extend(tmp2)
drop_col = list(set(drop_col))
zx_col = [i for i in zx_col if i not in drop_col]
df['rid'] = df.apply(lambda x: [ i+'x'+str(x[i]) for i in zx_col],axis=1) # 行为序列
df = w2v_feat(df)
del df['rid']
gc.collect()
```
## <span id='21'>黄金组合特征</span>
```python
for cate in ['民宿周边','邮编']:
    for f in ['评论间隔_day']:
        df['{}_{}_mean'.format(cate,f)] = df.groupby(cate)[f].transform('mean')
        df['{}_{}_median'.format(cate,f)] = df.groupby(cate)[f].transform('median')
        df['{}_div_{}_{}_mean'.format(f,cate,f)] = df[f]/(df['{}_{}_mean'.format(cate,f)]+1e-5)
        df['{}_div_{}_{}_median'.format(f,cate,f)] = df[f]/(df['{}_{}_median'.format(cate,f)]+1e-5)
        df['{}_minus_{}_{}_mean'.format(f,cate,f)] = df[f] - df['{}_{}_mean'.format(cate,f)]
```
## <span id='22'>数据倾斜</span>
```python
df['skew_A_B_1'] = df['A_B_median'] - df['A_B_mean']
df['skew_A_B_2'] = df['skew_A_B_1'].map(abs)
df['skew_A_B_ratio'] = df['A_B_median'] / (df['A_B_mean']+1e-5)

#变异系数
df['A_B_cv'] =  df['A_B_std'] / (df['A_B_mean']+1e-5)
```
