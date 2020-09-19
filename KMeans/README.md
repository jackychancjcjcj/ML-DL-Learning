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
