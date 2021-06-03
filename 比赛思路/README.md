## 时间序列
* 考虑差分的时候，会联想到最开始时间的数据无法和再上一段时间数据差分，因此可以考虑要不要包含最开始时间的数据。总结来说是，考虑训练集包含的样本。

## 线上线下
* 不能光以线上指标作为衡量，因为可能会出现数据穿越的情况，导致线上分数很高。

## 回归问题
把预测结果固定一下 下值取=0.025的值，上值取=0.975的值
```python
import numpy as np
low_bound,high_bound = df[target].quantile(0.025),df[target].quantile(0.975)
np.clip(y_pred,low_bound,high_bound)
```
