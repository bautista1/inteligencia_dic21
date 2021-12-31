import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np


x=np.array([
    [8,2],
[9,7],
[2,12],
[9,1],
[10,7],
[3,14],
[8,1],
[1,13]
])

kmeans=KMeans(n_clusters=3)
kmeans.fit(x)

print(kmeans.cluster_centers_)
plt.scatter(x[:,0],x[:,1],c=kmeans.labels_,cmap='raibow')
plt.show()