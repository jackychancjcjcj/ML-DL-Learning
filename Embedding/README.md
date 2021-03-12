# Embedding（实体嵌入）的理解
`Embedding`是将离散变量序列化为连续变量的方法。一般可用于NLP的word embedding和类别数据的entity embedding。在数据挖掘场景下，当分类特征水平很高的时候，one-hot编码经常会带来维度爆炸的问题，并且会丢失类别间潜在的关联。  
Word Embedding有的时候也被称作为分布式语义模型或向量空间模型等,所以从名字和其转换的方式我们就可以明白, Word Embedding技术可以将相同类型的词归到一起,例如苹果，芒果香蕉等，在投影之后的向量空间距离就会更近，而书本，房子这些则会与苹果这些词的距离相对较远。  
* 实体嵌入可以用低维度的连续空间表示高维的离散空间，通过嵌入层处理。  
* 实体嵌入通过将分类属性放入网络的全连接层的输入单元中，后接几个单元数较输入层更少的隐藏层（连续型变量直接接入第一个隐藏层），经过神经网络训练后输出第一个隐藏层中分类变量关联的隐层单元，作为提取的特征，用于各种模型的输入。
## Embedding目的
目前为止，Word Embedding可以用到特征生成，文件聚类，文本分类和自然语言处理等任务，例如：
* 计算相似的词：Word Embedding可以被用来寻找与某个词相近的词。
* 构建一群相关的词：对不同的词进行聚类，将相关的词聚集到一起；
* 用于文本分类的特征：在文本分类问题中，因为词没法直接用于机器学习模型的训练，所以我们将词先投影到向量空间,这样之后便可以基于这些向量进行机器学习模型的训练；
* 用于文件的聚类    
## 为什么使用word2vec？
不管是Countervector还是TFIDF,我们发现它们都是从全局词汇的分布来对文本进行表示,所以缺点也明显,它忽略了单个文本句子中词的顺序, 例如 'this is bad' 在BOW中的表示和 'bad is this'是一样的;它忽略了词的上下文,假设我们写一个句子,"He loved books. Education is best found in books".我们会在处理这两句话的时候是不会考虑前一个句子或者后一个句子是什么意思，但是他们之间是存在某些关系的。为了克服上述的两个缺陷，Word2Vec被开发出来并用来解决上述的两个问题。  
## 优缺点
* 这种针对分类变量特征提取的方法对于提升预测准确率来说是很有效的，跟模型stacking一样有效。  
* 关于类别的embedding，一般比较常用的是item2vec的思路，就是用word2vec来处理多值离散特征，需要注意的是，只有类别之间存在关联性，比如特征A的类别是男，女，特征B的类别是包包，鞋子，剃须刀之类的，存在共现的情况才可以取使用word2vec的思路来做自监督的embedding  
* 当然这种方法不是很稳定，如果类别特征之间独立性较强，用word2vec的方式没什么作用。
## 目录
* [TF-IDF](#1)
* [Word2Vec](#2)
* [jieba分词](#3)
# <span id='1'>TF-IDF</span>
`TF-IDF`(Term Frequency-inverse Document Frequency)是一种针对关键词的统计分析方法，用于评估一个词对一个文件集或者一个语料库的重要程度。一个词的重要程度跟它在文章中出现的次数成正比，跟它在语料库出现的次数成反比。这种计算方式能有效避免常用词对关键词的影响，提高了关键词与文章之间的相关性。  
* 其中TF指的是某词在文章中出现的总次数，该指标通常会被归一化定义为TF=（某词在文档中出现的次数/文档的总词量），这样可以防止结果偏向过长的文档（同一个词语在长文档里通常会具有比短文档更高的词频）。
* IDF逆向文档频率，包含某词语的文档越少，IDF值越大，说明该词语具有很强的区分能力，IDF=loge（语料库中文档总数/包含该词的文档数+1），+1的原因是避免分母为0。TF-IDF=TF x IDF，TF-IDF值越大表示该特征词对这个文本的重要性越大。
## 示例
### demo-1
```python
from sklearn.feature_extraction.text import TfidfVectorizer
corpus = [
    'This is the first document.',
    'This document is the second document.',
    'And this is the third one.',
    'Is this the first document?',
]
vectorizer = TfidfVectorizer()
# 转化为tf-idf
tdm = vectorizer.fit_transform(corpus)
print(tdm.toarray())
space = vectorizer.vocabulary_
print(space)
```
### demo-2
```python
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
corpus = [
    'This is the first document.',
    'This document is the second document.',
    'And this is the third one.',
    'Is this the first document?',
]
vectorizer = CountVectorizer()
transformer = TfidfTransformer()
tdm = tranformer.fit_transform(vectorizer.fit_transform(corpus))
tdm.toarray()
```
demo-1和demo-2的作用是一样的
### 实际应用
```python
# tfidif 处理经营范围的特征
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
#cn_stopwords.txt来源于 https://github.com/goto456/stopwords
def stopwordslist():
    stopwords = [line.strip() for line in open(r'E:\DL-ML\CCF企业非法集资风险预测\cn_stopwords.txt',encoding='UTF-8').readlines()]
    return stopwords
# 创建一个停用词列表
stopwords = stopwordslist()
stopwords+=['、', '；', '，', '）','（']
#
train_df_scope=base_info.merge(entprise_info)[['id','opscope','label']]
test_df_scope=base_info[base_info['id'].isin(entprise_evaluate['id'].unique().tolist())]
test_df_scope=test_df_scope.reset_index(drop=True)[['id','opscope']]
str_label_0=''
str_label_1=''
for index,name,opscope,label in train_df_scope.itertuples():
    # 结巴分词
    seg_text = jieba.cut(opscope.replace("\t", " ").replace("\n", " "))
    outline = " ".join(seg_text)
    out_str=""
    for per in outline.split():
        if per not in stopwords: 
            out_str += per
            out_str+=" "
    if label==0:
        str_label_0+=out_str
    else:
        str_label_1+=out_str
corpus=[str_label_0,str_label_1]
vectorizer=CountVectorizer()#该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
transformer=TfidfTransformer()#该类会统计每个词语的tf-idf权值
tfidf=transformer.fit_transform(vectorizer.fit_transform(corpus))#第一个fit_transform是计算tf-idf，第二个fit_transform是将文本转为词频矩阵
word=vectorizer.get_feature_names()#获取词袋模型中的所有词语总共7175个词语
weight=tfidf.toarray()#将(2, 7175)tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重
# for i in range(len(weight)):#打印每类文本的tf-idf词语权重，第一个for遍历所有文本，第二个for便利某一类文本下的词语权重
#     #
#     for j in range(len(word)):
#         print(word[j],weight[i][j])
#下面将会根据tfidi算出来的权重将经营范围的文本特征转换为数值(利用weight[1,:]也即各个词语在第二类(违法类中所占据的权重之和))
illegal_word_weights={}
for i in range(len(word)):
    illegal_word_weights[word[i]]=weight[1][i]
tfidi_opscope=[]
for index,name,opscope in base_info[['id','opscope']].itertuples():
    # 
    seg_text = jieba.cut(opscope.replace("\t", " ").replace("\n", " "))
    outline = " ".join(seg_text)
    tfidi_frt = 0
    for per in outline.split():
        if per in illegal_word_weights: 
            tfidi_frt+=illegal_word_weights[per]
    tfidi_opscope.append(tfidi_frt)
base_info['tfidif_opscope']=tfidi_opscope
print('对opscope提取tfidif特征完毕..........')
```
# <span id='2'>Word2Vec</span>
# <span id='3'>jieba分词</span>
得到要处理的字符串：（stopword_list可以网上搜）
```python
str_tmp=''
for i in range(len(df['opscope1'])):
    str_tmp=str_tmp+df['opscope1'][i]
stopword_list=stopword_list+['）','（','(',')','*','【','】',' ','，',',']  
```
得到处理后的结果：
```python
words_all=[i for i in jieba.cut(str_tmp) if i not in stopword_list]
tags1 = jieba.analyse.extract_tags(str_tmp, topK=300)
```
