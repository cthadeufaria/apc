import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def filter_10_nonzeros(a):
    diff = 10 - a[np.where(np.nan_to_num(a) != 0)].shape[0]
    if diff <= 0:
        return a[np.where(np.nan_to_num(a) != 0)][0:10]
    else:
        return np.concatenate((a[np.where(np.nan_to_num(a) != 0)], np.zeros(shape=(diff,))))

"""Preprocess data."""
def filter_nonzeros(a):
    return a[np.where(a != 0)]
df_test2obs = pd.read_csv('project02/data_test2obs.csv', index_col=0)
Lidar_range = df_test2obs.iloc[:, np.arange(2,362,1)].values
px = df_test2obs["px"].values
py = df_test2obs["py"].values
t1=1*10 # 1sec times number of samples/second
t2=30*10
angle = np.linspace(-179, 180, num=360)

# Clean outliers greater than 5m
Lidar_range[np.where(Lidar_range > 5.0)] = 0

# Build the cloud points in 2D plan
xy = np.zeros(shape=(*Lidar_range.shape, 2))
for t in range(len(Lidar_range)):
    for i in range(len(Lidar_range[t])):
        if Lidar_range[t][i] > 0.:
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
                new_centroids[l] = [np.random.uniform(min(X[:, 0]), max(X[:, 0])), np.random.uniform(min(X[:, 1]), max(X[:, 1]))]
        if np.count_nonzero(centroids - new_centroids) == 0:
            diff = 0
        else: 
            centroids = new_centroids
            centroids_list.append(centroids)

    return centroids, cluster


centroids1, cluster1 = k_means(xy1, 1)
centroids2, cluster2 = k_means(xy2, 2)

"""Print clusters and centroids fot t = 1s and t = 32s."""
plt.scatter(xy1[:, 0], xy1[:, 1], c='blue')
plt.scatter(xy2[:, 0], xy2[:, 1], c='yellow')
plt.scatter(centroids1[:, 0], centroids1[:, 1], c='black', s=200, alpha=0.3)
plt.scatter(centroids2[:, 0], centroids2[:, 1], c='orange', s=200, alpha=0.3)
plt.show()

"""Estimate best number of clusters."""
def SSE(X, centroids, cluster):
    total = 0
    for i, point in enumerate(X):
        total += np.sqrt((centroids[int(cluster[i]), 0] - point[0]) ** 2 + (centroids[int(cluster[i]), 1] - point[1]) ** 2)
    return total

def find_bestk(derivatives, derivatives_range, threshold):
    bestk = 0
    for i in reversed(derivatives_range):
        if derivatives[i-1] < threshold:
            bestk = i+1
            break
    if bestk == 0:
        return 1
    elif bestk > 2:
        return 2
    return bestk

bestk = []
k_range = range(1, 9)
threshold = -100
for t in range(xy.shape[0]):
    cost_list = []
    for k in k_range:
        centroids, cluster = k_means(xy[t, :, :], k)
        cost = SSE(xy[t, :, :], centroids, cluster)
        cost_list.append(cost)
    derivatives = []
    derivatives_range = range(1, max(k_range))
    for i in derivatives_range:
        derivatives.append(cost_list[i] - cost_list[i-1])
    bestk.append(find_bestk(derivatives, derivatives_range, threshold))

plt.plot(k_range, cost_list, c='blue', marker='o')
plt.plot(derivatives_range, derivatives, c='orange', marker='*')
plt.legend(["Sum of Squared Error", "SSE 1st Derivative"], loc ="lower right")
plt.title("Plot for Last Iteration of (x,y) Observations")
plt.xlabel("Number of Clusters")
plt.show()

"""Build new dataset with one object for each snapshot."""
# xy2 = np.zeros(shape=(sum(bestk), *xy.shape[1:]))
xy2_list = []
for t in range(xy.shape[0]):
    cluster = k_means(xy[t, :, :], bestk[t])[1]
    for k in range(0, bestk[t]):
        xy2_sublist = []
        for feature in range(0, xy[t, :, :].shape[0]):
            if k == 0.:
                key = int(not(cluster[feature]))
            elif k == 1.:
                key = int(cluster[feature])
            xy2_sublist.append(xy[t, :, :][feature] * key)
        xy2_list.append(np.array(xy2_sublist))
        # xy2[k] = np.array(xy2_sublist)
xy_full = np.array(xy2_list)

"""Test ANN. Classify train dataset."""
lidar_range_full_10 = np.zeros(shape=(xy_full.shape[0:2]))
for t in range(xy_full.shape[0]):
    for i in range(xy_full.shape[1]):
        lidar_range_full_10[int(t)][int(i)] = (xy_full[t, i, 0] / np.cos(angle[i]/180*np.pi)) + (xy_full[t, i, 1] / np.sin(angle[i]/180*np.pi))
lidar_nonzeros = np.apply_along_axis(filter_10_nonzeros, 1, lidar_range_full_10)
print(lidar_nonzeros)
# classifications_ANN_full = np.zeros(shape=(len(lidar_nonzeros)))
# for i in range(lidar_nonzeros.shape[0]):
#     classifications_ANN_full[i] = model.feedforward(lidar_nonzeros[i])
# # Set classifications to discrete values.
# classifications_ANN_full = np.where(classifications_ANN_full >= 0.5, 1., 0.)