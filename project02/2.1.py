import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def filter_10_nonzeros_2(m):
    a = np.zeros(shape=m.shape)
    for c in range(m.shape[1]):
        a[c] = m[np.where(m > 0)][0:10]
    return a

def filter_nonzeros(a):
    return a[np.where(a != 0)]

"""Preprocess data."""
df_test2obs = pd.read_csv('project02/data_test2obs.csv', index_col=0)
Lidar_range = df_test2obs.iloc[:, np.arange(2,362,1)].values
px = df_test2obs["px"].values
py = df_test2obs["py"].values
t1=1*10 #1sec times number of samples/second
t2=32*10 #5sec times number of samples/second
angle = np.linspace(-179, 180, num=360)

# Build the cloud points in 2D plan
xy = np.zeros(shape=(*Lidar_range.shape, 2))
for t in range(len(Lidar_range)):
    for i in range(len(Lidar_range[t])):
        if Lidar_range[t][i] > 0:
            xy[t][i][0] = px[t]+Lidar_range[t][i]*np.cos(angle[i]/180*np.pi)
            xy[t][i][1] = py[t]+Lidar_range[t][i]*np.sin(angle[i]/180*np.pi)
xy1 = xy[t1, :, :]
xy2 = xy[t2, :, :]

xy1 = np.apply_along_axis(filter_nonzeros, 0, xy1)
xy2 = np.apply_along_axis(filter_nonzeros, 0, xy2)

"""K-means method."""
def k_means(X, k):
    cluster = np.zeros(shape=X.shape[0])
    centroids_0 = []
    for c in range(k):
        centroids_0.append([np.random.uniform(low=min(X[:, 0]), high=max(X[:, 0])), np.random.uniform(min(X[:, 1]), max(X[:, 1]))])

    diff = 1
    centroids = centroids_0
    centroids_list = [centroids]

    while diff:
        for i, point in enumerate(X):
            min_distance = np.inf
            for j, centroid in enumerate(centroids):
                distance = np.sqrt((centroid[0] - point[0]) ** 2 + (centroid[1] - point[1]) ** 2)
                if min_distance > distance:
                    min_distance = distance
                    cluster[i] = j
        total = np.zeros(shape=(k, 2))
        count = np.zeros(shape=(k))
        for i, point in enumerate(X):
            total[int(cluster[i])] += X[i]
            count[int(cluster[i])] += 1
        new_centroids = np.zeros(shape=(k, 2))
        for l in range(k):
            if count[l] > 0:
                new_centroids[l] = total[l] / count[l]
            else:
                new_centroids[k] = [np.random.uniform(min(X[:, 0]), max(X[:, 0])), np.random.uniform(min(X[:, 1]), max(X[:, 1]))]
        if np.count_nonzero(centroids - new_centroids) == 0:
            diff = 0
        else: 
            print(centroids - new_centroids)
            centroids = new_centroids
            centroids_list.append(centroids)

    return centroids, cluster


centroids1, cluster1 = k_means(xy1, 1)
centroids2, cluster2 = k_means(xy2, 2)

"""Print clusters and centroids."""
plt.scatter(xy1[:, 0], xy1[:, 1], c='blue')
plt.scatter(xy2[:, 0], xy2[:, 1], c='yellow')
plt.scatter(centroids1[:, 0], centroids1[:, 1], c='black', s=200, alpha=0.3)
plt.scatter(centroids2[:, 0], centroids2[:, 1], c='orange', s=200, alpha=0.3)
plt.show()