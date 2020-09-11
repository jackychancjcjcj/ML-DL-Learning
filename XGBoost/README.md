# XGBoost
![image](https://github.com/jackychancjcjcj/ML-DL-Learning/blob/master/framework.jpg)
## 简介
XGBoost的全称是eXtreme Gradient Boosting，它是经过优化的分布式梯度提升库，旨在高效、灵活且可移植。XGBoost是大规模并行boosting tree的工具，它是目前最快最好的开源 boosting tree工具包，比常见的工具包快10倍以上。在数据科学方面，有大量的Kaggle选手选用XGBoost进行数据挖掘比赛，是各大数据科学比赛的必杀武器；在工业界大规模数据方面，XGBoost的分布式版本有广泛的可移植性，支持在Kubernetes、Hadoop、SGE、MPI、 Dask等各个分布式环境上运行，使得它可以很好地解决工业界大规模数据的问题。  
## 与GBDT区别联系
* GBDT是机器学习算法，XGBoost是该算法的工程实现。
* 正则项：在使用CART作为基分类器时，XGBoost显式地加入了正则项来控制模型的复杂度，有利于防止过拟合，从而提高模型的泛化能力。
* 导数信息：GBDT在模型训练时只使用了代价函数的一阶导数信息，XGBoost对代价函数进行二阶泰勒展开，可以同时使用一阶和二阶导数。
* 基分类器：传统的GBDT采用CART作为基分类器，XGBoost支持多种类型的基分类器，比如线性分类器。
* 子采样：传统的GBDT在每轮迭代时使用全部的数据，XGBoost则采用了与随机森林相似的策略，支持对数据进行采样。
* 缺失值处理：传统GBDT没有设计对缺失值进行处理，XGBoost能够自动学习出缺失值的处理策略。
* 并行化：传统GBDT没有进行并行化设计，注意不是tree维度的并行，而是特征维度的并行。XGBoost预先将每个特征按特征值排好序，存储为块结构，分裂结点时可以采用多线程并行查找每个特征的最佳分割点，极大提升训练速度。
## 优缺点
### 优点
* 精度更高：GBDT 只用到一阶泰勒展开，而 XGBoost 对损失函数进行了二阶泰勒展开。XGBoost 引入二阶导一方面是为了增加精度，另一方面也是为了能够自定义损失函数，二阶泰勒展开可以近似大量损失函数；
灵活性更强：GBDT 以 CART 作为基分类器，XGBoost 不仅支持 CART 还支持线性分类器，使用线性分类器的 XGBoost 相当于带 L1 和 L2 正则化项的逻辑斯蒂回归（分类问题）或者线性回归（回归问题）。此外，XGBoost 工具支持自定义损失函数，只需函数支持一阶和二阶求导；
* 正则化：XGBoost 在目标函数中加入了正则项，用于控制模型的复杂度。正则项里包含了树的叶子节点个数、叶子节点权重的 L2 范式。正则项降低了模型的方差，使学习出来的模型更加简单，有助于防止过拟合，这也是XGBoost优于传统GBDT的一个特性。
* Shrinkage（缩减）：相当于学习速率。XGBoost 在进行完一次迭代后，会将叶子节点的权重乘上该系数，主要是为了削弱每棵树的影响，让后面有更大的学习空间。传统GBDT的实现也有学习速率；
* 列抽样：XGBoost 借鉴了随机森林的做法，支持列抽样，不仅能降低过拟合，还能减少计算。这也是XGBoost异于传统GBDT的一个特性；
* 缺失值处理：对于特征的值有缺失的样本，XGBoost 采用的稀疏感知算法可以自动学习出它的分裂方向；
* XGBoost工具支持并行：boosting不是一种串行的结构吗?怎么并行的？注意XGBoost的并行不是tree粒度的并行，XGBoost也是一次迭代完才能进行下一次迭代的（第t次迭代的代价函数里包含了前面t-1次迭代的预测值）。XGBoost的并行是在特征粒度上的。我们知道，决策树的学习最耗时的一个步骤就是对特征的值进行排序（因为要确定最佳分割点），XGBoost在训练之前，预先对数据进行了排序，然后保存为block结构，后面的迭代中重复地使用这个结构，大大减小计算量。这个block结构也使得并行成为了可能，在进行节点的分裂时，需要计算每个特征的增益，最终选增益最大的那个特征去做分裂，那么各个特征的增益计算就可以开多线程进行。
* 可并行的近似算法：树节点在进行分裂时，我们需要计算每个特征的每个分割点对应的增益，即用贪心法枚举所有可能的分割点。当数据无法一次载入内存或者在分布式情况下，贪心算法效率就会变得很低，所以XGBoost还提出了一种可并行的近似算法，用于高效地生成候选的分割点。
### 缺点
* 虽然利用预排序和近似算法可以降低寻找最佳分裂点的计算量，但在节点分裂过程中仍需要遍历数据集；
* 预排序过程的空间复杂度过高，不仅需要存储特征值，还需要存储特征对应样本的梯度统计值的索引，相当于消耗了两倍的内存。
### 为什么采用二阶导
* 二阶信息本身就能让梯度收敛更快更准确。这一点在优化算法里的牛顿法中已经证实。可以简单认为一阶导指引梯度方向，二阶导指引梯度方向如何变化。简单来说，相对于GBDT的一阶泰勒展开，XGBoost采用二阶泰勒展开，可以更为精准的逼近真实的损失函数。
* Xgboost官网上有说，当目标函数是MSE时，展开是一阶项（残差）+二阶项的形式（官网说这是一个nice form），而其他目标函数，如logloss的展开式就没有这样的形式。为了能有个统一的形式，所以采用泰勒展开来得到二阶项，这样就能把MSE推导的那套直接复用到其他自定义损失函数上。简短来说，就是为了统一损失函数求导的形式以支持自定义损失函数。这是从为什么会想到引入泰勒二阶的角度来说的。
### 如何处理缺失值
* 在特征k上寻找最佳 split point 时，不会对该列特征 missing 的样本进行遍历，而只对该列特征值为 non-missing 的样本上对应的特征值进行遍历，通过这个技巧来减少了为稀疏离散特征寻找 split point 的时间开销。
* 在逻辑实现上，为了保证完备性，会将该特征值missing的样本分别分配到左叶子结点和右叶子结点，两种情形都计算一遍后，选择分裂后增益最大的那个方向（左分支或是右分支），作为预测时特征值缺失样本的默认分支方向。
* 如果在训练中没有缺失值而在预测中出现缺失，那么会自动将缺失值的划分方向放到右子结点。
### 参数调节
#### 总览
    from xgboost import XGBClassifier
    clf = XGBClassifier(max_depth=3, learning_rate=0.1, n_estimators=100, silent=True, objective="binary:logistic", booster='gbtree', n_jobs=1, nthread=None, gamma=0,
                    min_child_weight=1, max_delta_step=0, subsample=1, colsample_bytree=1, colsample_bylevel=1, reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
                    base_score=0.5, random_state=0, seed=None, missing=None, **kwargs)
    clf.fit(x, y, sample_weight=None, eval_set=None, eval_metric=None, early_stopping_rounds=None, verbose=True, xgb_model=None, sample_weight_eval_set=None, callbacks=None)
XGBoost的参数分为三大类：
* General Parameters: 控制总体的功能
* Booster Parameters: 控制单个学习器的属性
* Learning Task Parameters: 控制调优的步骤
#### General Parameters:
    booster [default=gbtree]
    a: 表示应用的弱学习器的类型, 推荐用默认参数
    b: 可选的有gbtree, dart, gblinear
        gblinear是线性模型，表现很差，接近一个LASSO
        dart是树模型的一种，思想是每次训练新树的时候，随机从前m轮的树中扔掉一些，来避免过拟合
        gbtree即是论文中主要讨论的树模型，推荐使用
    silent [default=0]:
    设为1则不打印执行信息，设为0打印信息 a: 不推荐使用，推荐使用verbosity参数来代替，功能更强大
    verbosity [default: 1]
    a: 训练过程中打印的日志等级，0 (silent), 1 (warning), 2 (info), 3 (debug)
    nthread [default to maximum number of threads available if not set]
    这个是设置并发执行的信息，设置在几个核上并发如果你希望在机器的所有可以用的核上并发执行，则采用默认的参数
    a: 训练过程中的并行线程数
    b: 如果用的是sklearn的api，那么使用n_jobs来代替
#### Booster Parameters
    n_estimators 总共迭代的次数，即决策树的个数
    learning_rata [default=0.3] (以前是用eta的，后面已经改了）通过在每一步中缩小权重来让模型更加鲁棒一般常用的数值: 0.01-0.2
    min_child_weight [default=1] 这个参数用来控制过拟合Too high values can lead to under-fitting hence, it should be tuned using CV.
    a: 最小的叶子节点权重
    b: 在普通的GBM中，叶子节点样本没有权重的概念，其实就是等权重的，也就相当于叶子节点样本个数
    c: 越小越没有限制，容易过拟合，太高容易欠拟合
    max_depth [default=6] The maximum depth of a tree, same as GBM. 控制子树中样本数占总的样本数的最低比例控制过拟合，如果树的深度太大会导致过拟合应该使用CV来调节。Typical values: 3-10
    a: 树的最大深度
    b: 这个值对结果的影响算是比较大的了，值越大，树的深度越深，模型的复杂度就越高，就越容易过拟合
    c: 注意如果这个值被设置的较大，会吃掉大量的内存
    d: 一般来说比价合适的取值区间为[3, 10]
    max_leaf_nodes 叶子节点的最大值，也是为了通过树的规模来控制过拟合。如果叶子树确定了，对于2叉树来说高度也就定了，此时以叶子树确定的高度为准
    gamma [default=0] 如果分裂能够使loss函数减小的值大于gamma，则这个节点才分裂。gamma设置了这个减     小的最低阈值。如果gamma设置为0，表示只要使得loss函数减少，就分裂。这个值会跟具体的loss     函数相关，需要调节。
    max_delta_step [default=0] 如果参数设置为0，表示没有限制。如果设置为一个正值，会使得更新步更加谨慎。不是很经常用，但是在逻辑回归时候，使用它可以处理类别不平衡问题。
    a: 适用于正负样本不均衡情况下，控制学习速率(类似eta)最大为某个值，不能超过这个阈值
    b: 首先我们有参数eta来控制学习速率，为了后面学习到更多，每一步在权重上乘上这个因子，降低速度
    c: 但是在正负样本不均衡的情况下eta不足够，因为此时由于二阶导接近于0的原因，权重会特别大
    d: 这个参数就是用来控制学习速率最大不能超过这个数值
    subsample [default=1] 对原数据集进行随机采样来构建单个树。这个参数代表了在构建树时候 对原数据集采样的百分比。eg：如果设为0.8表示随机抽取样本中80%的个体来构建树。相对小点的数值可以防止过拟     合，但是过小的数值会导致欠拟合（因为采样过小）。 一般取值 0.5 到 1
    colsample_bytree [default=1] 创建树的时候，从所有的列中选取的比例。e.g：如果设为0.8表示随机抽取80%的列 用来创建树Typical values: 0.5-1
    colsample_bylevel [default=1] 每一层深度的树特征抽样比例
    colsample_bynode 每一个节点的特征抽样比例
    lambda [default: 1] a: 损失函数中的L2正则化项的系数，类似RidgeRegression，减轻过拟合
    alpha [default: 0] a: 损失函数中的L1正则化项的系数，类似LASSO，减轻过拟合
    scale_pos_weight [default: 1] a: 在正负样本不均衡的情况下，此参数需要设置，通常为: sum(负样本) / sum(正样本)
#### Learning Task Parameters
    objective [default: reg:squarederror(均方误差)]
    a: 目标函数的选择，默认为均方误差损失，当然还有很多其他的，这里列举几个主要的
    b: reg:squarederror       均方误差
    c: reg:logistic           对数几率损失，参考对数几率回归(逻辑回归)
    d: binary:logistic        二分类对数几率回归，输出概率值
    e: binary:hinge           二分类合页损失，此时不输出概率值，而是0或1
    f: multi:softmax          多分类softmax损失，此时需要设置num_class参数
    eval_metric [default: 根据objective而定]
    a: 模型性能度量方法，主要根据objective而定，也可以自定义一些，下面列举一些常见的
    b: rmse : root mean square error     也就是平方误差和开根号
    c: mae  : mean absolute error        误差的绝对值再求平均
    d: auc  : area under curve           roc曲线下面积
    e: aucpr: area under the pr curve    pr曲线下面积
    num_boost_round
    a: 迭代次数，这货其实跟sklearn中的n_estimators是一样的
    b: sklearn的api中用n_estimators，原始xgb中用num_boost_round
    evals
    a: 训练过程中通过计算验证集的指标，观察模型性能的数据集
    b: 指标就是通过eval_metric参数来制定的
    early_stopping_rounds
    a: 在num_boost_round的轮训练中，如果过程中指标经过early_stopping_rounds轮还没有减少那么就停止训练
    b: 指标是通过evals的验证集，计算eval_metric的指标
    



## 参考资料
[XGBoost原文-陈天奇](https://github.com/jackychancjcjcj/ML-DL-Learning/blob/master/XGBoost.pdf)  
[1.Introduction to Boosted Tree](https://github.com/jackychancjcjcj/ML-DL-Learning/blob/master/BoostedTree.pdf)  
[2.GBDT算法原理及系统设计](https://github.com/jackychancjcjcj/ML-DL-Learning/blob/master/gbdt.pdf)  
[3.深入理解XGBoost-知乎](https://zhuanlan.zhihu.com/p/83901304)  
