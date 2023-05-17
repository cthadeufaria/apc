import numpy as np
import pandas as pd
import os

def EuclideanDistance(test, train):
    x = len(test[0])
    train_length = len(train)
    distances = []
    for j in range(train_length):
        obs = len(train[j][0])
        distance = 0
        for i in range(x):
            for o in range(obs):
                distance += np.sqrt((test[0][i] - train[j][0][o]) ** 2 + (test[1][i] - train[j][1][o]) ** 2)
        distances.append(distance)
    return distances

def KNNclassifier(k, x, data, classes):
    dist = EuclideanDistance(x, data)
    ind = np.argsort(np.array(dist), axis=0)
    classes_sorted = classes[ind]
    c1 = 0
    c2 = 0
    for i in range(k):
        if classes_sorted[i] == 1:
            c1 += 1
        else:
            c2 += 1
    if c2 > c1:
        return 2
    else:
        return 1

print(os.getcwd())
df_train = pd.read_csv('project02/data_train.csv', index_col=0)
df_test = pd.read_csv('project02/data_test.csv', index_col=0)

# df_train = pd.read_csv('data_train.csv', index_col=0)
lidar_range_train = df_train.values[:, 2:362]
px_train = df_train["px"].values
py_train = df_train["py"].values
t_range_train = range(len(df_train))
angle = np.linspace(-179, 180, num=360)
x_y_train = []
for t in t_range_train:
    x_train, y_train = [], []
    for i in range(0, lidar_range_train.shape[1]):
        if lidar_range_train[t][i] > 0:
            x_train.append(px_train[t]+lidar_range_train[t][i]*np.cos(angle[i]/180*np.pi))
            y_train.append(py_train[t]+lidar_range_train[t][i]*np.sin(angle[i]/180*np.pi))
    x_y_train.append([x_train, y_train])

# df_test = pd.read_csv('data_test.csv', index_col=0)
lidar_range_test = df_test.values[:, 2:362]
px_test = df_test["px"].values
py_test = df_test["py"].values
t_range_test = range(len(df_test))
angle = np.linspace(-179, 180, num=360)
x_y_test = []
for t in t_range_test:
    x_test, y_test = [], []
    for i in range(0, lidar_range_test.shape[1]):
        if lidar_range_test[t][i] > 0:
            x_test.append(px_test[t]+lidar_range_test[t][i]*np.cos(angle[i]/180*np.pi))
            y_test.append(py_test[t]+lidar_range_test[t][i]*np.sin(angle[i]/180*np.pi))
    x_y_test.append([x_test, y_test])

# Classify test dataset using train dataset
k_list = [1, 3, 5, 7, 9]
classes_train = df_train.values[:,-1]
classes_test = df_test.values[:, -1]
classifications = np.zeros(shape=(len(x_y_test), len(k_list)))
for k in range(len(k_list)):
    for i in range(len(x_y_test)):
        # classifications.append(KNNclassifier(k, x_y_test[i], x_y_train, classes_train))
        classifications[i, k] = KNNclassifier(k, x_y_test[i], x_y_train, classes_train)
        print("classification ", i, " for k = ", k_list[k], " = ", classifications[i][k])
print(classifications)

accuracy = dict()
for k in range(len(k_list)):
    accuracy[k_list[k]] = sum(classifications[:,k] == classes_test)/len(classes_test)