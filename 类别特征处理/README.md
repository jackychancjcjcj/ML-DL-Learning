# 写在前面
类别型特征（`categorical feature`）主要是指职业，血型等在有限类别内取值的特征。它的原始输入通常是字符串形式，大多数算法模型不接受数值型特征的输入，针对数值型的类别特征会被当成数值型特征，从而造成训练的模型产生错误。
# 文章目录
* [1.Label encoding](#1)  
* [2.序列编码(Ordinal Encoding)](#2)
* [3.独热编码(One-Hot Encoding)](#3)
* [4.频数编码(Frequency Encoding/Count Encoding)](#4)
* [5.目标编码(Target Encoding/Mean Encoding)](#5)
* [6.Beta Target Encoding]
* [7.M-Estimate Encoding]
* [8.James-Stein Encoding]
* [9.Weight of Evidence Encoder]
* [10.Leave-one-out Encoder (LOO or LOOE)]
* [11.Binary Encoding]
* [12.Hashing Encoding]
* [13.Probability Ratio Encoding]
* [14.Sum Encoder (Deviation Encoder, Effect Encoder)]
* [15.Helmert Encoding]
* [16.CatBoost Encoding]
# 参考文章
[kaggle竞赛之类别特征处理](https://mp.weixin.qq.com/s/qPlh6Gbb-ZvZrESM6e2ZSA)
## <span id="1">1.Label encoding</span>
`Label Encoding`是使用字典的方式，将每个类别标签与不断增加的整数相关联，即生成一个名为class_的实例数组的索引。  
Scikit-learn中的LabelEncoder是用来对分类型特征值进行编码，即对不连续的数值或文本进行编码。其中包含以下常用方法：
* fit(y) ：fit可看做一本空字典，y可看作要塞到字典中的词。
* fit_transform(y)：相当于先进行fit再进行transform，即把y塞到字典中去以后再进行transform得到索引值。
* inverse_transform(y)：根据索引值y获得原始数据。
* transform(y) ：将y转变成索引值
```python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
city_list = ["paris", "paris", "tokyo", "amsterdam"]
le.fit(city_list)
print(le.classes_)  # 输出为：['amsterdam' 'paris' 'tokyo']
city_list_le = le.transform(city_list)  # 进行Encode
print(city_list_le)  # 输出为：[1 1 2 0]
city_list_new = le.inverse_transform(city_list_le)  # 进行decode
print(city_list_new) # 输出为：['paris' 'paris' 'tokyo' 'amsterdam']
```
## <span id='2'>2.序列编码(Ordinal Encoding)</span>
Ordinal Encoding即最为简单的一种思路，对于一个具有m个category的Feature，我们将其对应地映射到 [0,m-1] 的整数。当然 Ordinal Encoding 更适用于 Ordinal Feature，即各个特征有内在的顺序。例如对于”学历”这样的类别，”学士”、”硕士”、”博士” 可以很自然地编码成 [0,2]，因为它们内在就含有这样的逻辑顺序。但如果对于“颜色”这样的类别，“蓝色”、“绿色”、“红色”分别编码成[0,2]是不合理的，因为我们并没有理由认为“蓝色”和“绿色”的差距比“蓝色”和“红色”的差距对于特征的影响是不同的。  
```python
ord_map = {'Gen 1': 1, 'Gen 2': 2, 'Gen 3': 3, 'Gen 4': 4, 'Gen 5': 5, 'Gen 6': 6}
df['GenerationLabel'] = df['Generation'].map(gord_map)
```
## <span id='3'>3.独热编码(One-Hot Encoding)</span>
在实际的机器学习的应用任务中，特征有时候并不总是连续值，有可能是一些分类值，如性别可分为male和female。在机器学习任务中，对于这样的特征，通常我们需要对其进行特征数字化，比如有如下三个特征属性：
* 性别：[“male”，”female”]
* 地区：[“Europe”，”US”，”Asia”]
* 浏览器：[“Firefox”，”Chrome”，”Safari”，”Internet Explorer”]
对于某一个样本，如[“male”，”US”，”Internet Explorer”]，我们需要将这个分类值的特征数字化，最直接的方法，我们可以采用序列化的方式：[0,1,3]。但是，即使转化为数字表示后，上述数据也不能直接用在我们的分类器中。因为，分类器往往默认数据是连续的，并且是有序的。按照上述的表示，数字并不是有序的，而是随机分配的。这样的特征处理并不能直接放入机器学习算法中。  
为了解决上述问题，其中一种可能的解决方法是采用独热编码（One-Hot Encoding）。独热编码，又称为一位有效编码。其方法是使用N位状态寄存器来对N个状态进行编码，每个状态都由他独立的寄存器位，并且在任意时候，其中只有一位有效。可以这样理解，对于每一个特征，如果它有m个可能值，那么经过独热编码后，就变成了m个二元特征。并且，这些特征互斥，每次只有一个激活。因此，数据会变成稀疏的。  
对于上述的问题，性别的属性是二维的，同理，地区是三维的，浏览器则是四维的，这样，我们可以采用One-Hot编码的方式对上述的样本[“male”，”US”，”Internet Explorer”]编码，male则对应着[1，0]，同理US对应着[0，1，0]，Internet Explorer对应着[0,0,0,1]。则完整的特征数字化的结果为：[1,0,0,1,0,0,0,0,1]。  
### 为什么能使用One-Hot Encoding？
* 使用one-hot编码，将离散特征的取值扩展到了欧式空间，离散特征的某个取值就对应欧式空间的某个点。在回归，分类，聚类等机器学习算法中，特征之间距离的计算或相似度的计算是非常重要的，而我们常用的距离或相似度的计算都是在欧式空间的相似度计算，计算余弦相似性，也是基于的欧式空间。  
* 将离散型特征使用one-hot编码，可以会让特征之间的距离计算更加合理。比如，有一个离散型特征，代表工作类型，该离散型特征，共有三个取值，不使用one-hot编码，计算出来的特征的距离是不合理。那如果使用one-hot编码，显得更合理。  
### 独热编码优缺点
* 优点：独热编码解决了分类器不好处理属性数据的问题，在一定程度上也起到了扩充特征的作用。它的值只有0和1，不同的类型存储在垂直的空间。
* 缺点：当类别的数量很多时，特征空间会变得非常大。在这种情况下，一般可以用PCA（主成分分析）来减少维度。而且One-Hot Encoding+PCA这种组合在实际中也非常有用。
### One-Hot Encoding使用场景
* 独热编码用来解决类别型数据的离散值问题。将离散型特征进行one-hot编码的作用，是为了让距离计算更合理，但如果特征是离散的，并且不用one-hot编码就可以很合理的计算出距离，那么就没必要进行one-hot编码，比如，该离散特征共有1000个取值，我们分成两组，分别是400和600,两个小组之间的距离有合适的定义，组内的距离也有合适的定义，那就没必要用one-hot 编码。
* 基于树的方法是不需要进行特征的归一化，例如随机森林，bagging 和 boosting等。对于决策树来说，one-hot的本质是增加树的深度，决策树是没有特征大小的概念的，只有特征处于他分布的哪一部分的概念。
### 基于Scikit-learn 的one hot encoding
`LabelBinarizer`：将对应的数据转换为二进制型，类似于onehot编码，这里有几点不同：
* 可以处理数值型和类别型数据
* 输入必须为1D数组
* 可以自己设置正类和父类的表示方式
```python
from sklearn.preprocessing import LabelBinarizer
 
lb = LabelBinarizer()
 
city_list = ["paris", "paris", "tokyo", "amsterdam"]
 
lb.fit(city_list)
print(lb.classes_)  # 输出为：['amsterdam' 'paris' 'tokyo']
 
city_list_le = lb.transform(city_list)  # 进行Encode
print(city_list_le)  # 输出为：
# [[0 1 0]
#  [0 1 0]
#  [0 0 1]
#  [1 0 0]]
 
city_list_new = lb.inverse_transform(city_list_le)  # 进行decode
print(city_list_new)  # 输出为：['paris' 'paris' 'tokyo' 'amsterdam']
```
OneHotEncoder只能对数值型数据进行处理，需要先将文本转化为数值（Label encoding）后才能使用，只接受2D数组
```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
df = pd.DataFrame([
    ['green', 'Chevrolet', 2017],
    ['blue', 'BMW', 2015],
    ['yellow', 'Lexus', 2018],
])
df.columns = ['color', 'make', 'year']
ohe = OneHotEncoder()
a = ohe.fit_transform(df)
result = pd.DataFrame(a.toarray(),columns=ohe.get_feature_names())
```
### 基于Pandas的one hot encoding
基于Pandas的one hot encoding
```python
import pandas as pd
 
df = pd.DataFrame([
    ['green', 'Chevrolet', 2017],
    ['blue', 'BMW', 2015],
    ['yellow', 'Lexus', 2018],
])
df.columns = ['color', 'make', 'year']
df_processed = pd.get_dummies(df, prefix_sep="_", columns=df.columns[:-1])
print(df_processed)
```
get_dummies的优势在于:
* 本身就是 pandas 的模块，所以对 DataFrame 类型兼容很好
* 不管你列是数值型还是字符串型，都可以进行二值化编码
* 能够根据指令，自动生成二值化编码后的变量名  
get_dummies虽然有这么多优点，但毕竟不是 sklearn 里的transformer类型，所以得到的结果得手动输入到 sklearn 里的相应模块，也无法像 sklearn 的transformer一样可以输入到pipeline中进行流程化地机器学习过程。
## <span id='4'>频数编码(Frequency Encoding/Count Encoding)</span>
将类别特征替换为训练集中的计数（一般是根据训练集来进行计数，属于统计编码的一种，统计编码，就是用类别的统计特征来代替原始类别，比如类别A在训练集中出现了100次则编码为100）。这个方法对离群值很敏感，所以结果可以归一化或者转换一下（例如使用对数变换）。未知类别可以替换为1。

频数编码使用频次替换类别。有些变量的频次可能是一样的，这将导致碰撞。尽管可能性不是非常大，没法说这是否会导致模型退化，不过原则上我们不希望出现这种情况。
```python
import pandas as pd
data_count = data.groupby('城市')['城市'].agg({'频数':'size'}).reset_index()
data = pd.merge(data, data_count, on = '城市', how = 'left')
```
## <span id='5'>5.目标编码(Target Encoding/Mean Encoding)</span>
目标编码（target encoding），亦称均值编码（mean encoding）、似然编码（likelihood encoding）、效应编码（impact encoding），是一种能够对高基数（high cardinality）自变量进行编码的方法 (Micci-Barreca 2001) 。  
如果某一个特征是定性的（categorical），而这个特征的可能值非常多（高基数），那么目标编码（Target encoding）是一种高效的编码方式。在实际应用中，这类特征工程能极大提升模型的性能。  
一般情况下，针对定性特征，我们只需要使用sklearn的OneHotEncoder或LabelEncoder进行编码。  
LabelEncoder能够接收不规则的特征列，并将其转化为从0到n-1的整数值（假设一共有n种不同的类别）；OneHotEncoder则能通过哑编码，制作出一个m\*n的稀疏矩阵（假设数据一共有m行，具体的输出矩阵格式是否稀疏可以由sparse参数控制）。  
定性特征的基数（cardinality）指的是这个定性特征所有可能的不同值的数量。在高基数（high cardinality）的定性特征面前，这些数据预处理的方法往往得不到令人满意的结果。  
高基数定性特征的例子：IP地址、电子邮件域名、城市名、家庭住址、街道、产品号码。  
```python
from category_encoders import TargetEncoder
import pandas as pd
from sklearn.datasets import load_boston
# prepare some data
bunch = load_boston()
y_train = bunch.target[0:250]
y_test = bunch.target[250:506]
X_train = pd.DataFrame(bunch.data[0:250], columns=bunch.feature_names)
X_test = pd.DataFrame(bunch.data[250:506], columns=bunch.feature_names)
# use target encoding to encode two categorical features
enc = TargetEncoder(cols=['CHAS', 'RAD'])
# transform the datasets
training_numeric_dataset = enc.fit_transform(X_train, y_train)
testing_numeric_dataset = enc.transform(X_test)
```
