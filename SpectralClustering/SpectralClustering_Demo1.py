import numpy as np
from sklearn import datasets
X, y = datasets.make_blobs(n_samples=500, n_features=6, centers=5, cluster_std=[0.4, 0.3, 0.4, 0.3, 0.4], random_state=11)

# 接着我们看看默认的谱聚类的效果：
from sklearn.cluster import SpectralClustering
y_pred = SpectralClustering().fit_predict(X)
from sklearn import metrics
# 输出的Calinski-Harabasz分数为：Calinski-Harabasz Score 14908.9325026
print("Calinski-Harabasz Score", metrics.calinski_harabaz_score(X, y_pred) )

# 由于我们使用的是高斯核，那么我们一般需要对n_clusters和gamma进行调参。选择合适的参数值。代码如下：
for index, gamma in enumerate((0.01,0.1,1,10)):
    for index, k in enumerate((3,4,5,6)):
        y_pred = SpectralClustering(n_clusters=k, gamma=gamma).fit_predict(X)
        print("Calinski-Harabasz Score with gamma=", gamma, "n_clusters=", k,"score:", metrics.calinski_harabaz_score(X, y_pred))