# 写在前面
类别型特征（`categorical feature`）主要是指职业，血型等在有限类别内取值的特征。它的原始输入通常是字符串形式，大多数算法模型不接受数值型特征的输入，针对数值型的类别特征会被当成数值型特征，从而造成训练的模型产生错误。
# 文章目录
* [1.Label encoding](#Label&nbsp;encoding)  
* [2.序列编码（Ordinal Encoding）]
* [3.独热编码(One-Hot Encoding)]
* [4.频数编码（Frequency Encoding/Count Encoding）]
* [5.目标编码（Target Encoding/Mean Encoding）]
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
## Label encoding
