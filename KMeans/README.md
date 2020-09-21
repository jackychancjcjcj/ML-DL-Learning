## sklearn.cluster.KMeans 参数
    class sklearn.cluster.KMeans(n_clusters=8, *, init='k-means++', n_init=10, max_iter=300, tol=0.0001,precompute_distances='deprecated', verbose=0, random_state=None, copy_x=True, n_jobs='deprecated', algorithm='auto')  
### Parameters
    n_clusters: int，默认是8  
        簇的个数    
    init: ['k-means++','random',ndarray], 默认是'k-means++'  
        k-means++:
        random: 根据`n_clusters`从数据中随机挑选  
        ndarray: 如果是传入ndarray的话，给定的是初始点的位置  
        初始点位选择的方法    
    n_init: int, default = 10  
        质心选择的次数，输出的是这么多次运行结果最好的，如果簇的个数比较多的话，这个值可以调大点。  
    max_iter: int, default = 300
        最大迭代次数，如果是凸数据集可以不管这个值，如果不是凸的话，可能很难收敛，  
    tol: float, default = 1e-4  
        一般不用改。
    verbose: int, default = 0  
    random_state: int, default = None  
        选择质心的随机种子。
    copy_x: bool, default = True  
    algorithm: ['auto','full','elkan'], default = 'auto'  
        auto: 根据数据值是否稀疏，稠密选择`elkan`，稀疏选择`full`  
### 优化
* 因为是基于距离的聚类方法，输入数据要缩放（标准化），不同维度差别过大的话，可能会造成少数变量过高的影响。
* 输入数据类型不同的话，部分是数值型部分是分类变量，需要做特别处理。（用独热编码进行处理，但会使得数据维度上升，但如果使用标签编码就无法很好的处理数据中的顺序。还有一种方法是对于数值类型变量和分类变量分开处理，并将结果结合起来，（如`k-mode`(https://github.com/nicodv/kmodes)和`k-prototype`）。
* 输出结果非固定，多次运行结果可能不同。（设定random state的话就可以固定）另外如果`k均值`一直在大幅度变化，可能就是数据不适合k-means方法。
* 数据量过大时可以考虑使用MiniBatchKMeans。
* 高维数据上并非所有维度都有意义，这种情况下`k均值`的结果往往不好，通过划分子空间的算法（sub-spacing method）效果可能会更好。
* 在数据量不大时，可以优先尝试其他算法。当数据量过大时，可以试试HDBSCAN。仅当数据量巨大且无法降维或降低数量时，尝试使用k均值。
