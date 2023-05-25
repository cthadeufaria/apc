import numpy as np
import pandas as pd
import os

def EuclideanDistance(test, train):
    # distances = []
    # for obs in train:
    #     distances.append(np.sqrt(sum((np.array(test) - np.array(obs))**2)))
    # return distances
    return [np.sqrt(sum((np.array(test) - np.array(observation))**2)) for observation in train]

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
# px_train = df_train["px"].values
# py_train = df_train["py"].values
# t_range_train = range(len(df_train))
# angle = np.linspace(-179, 180, num=360)
# x_y_train = []
# lidar_range_train_list = []
# for t in t_range_train:
#     x_train, y_train = [], []
#     lidar_range_train_sublist = []
#     for i in range(0, lidar_range_train.shape[1]):
#         if lidar_range_train[t][i] > 0:
#             x_train.append(px_train[t]+lidar_range_train[t][i]*np.cos(angle[i]/180*np.pi))
#             y_train.append(py_train[t]+lidar_range_train[t][i]*np.sin(angle[i]/180*np.pi))
#             lidar_range_train_sublist.append(lidar_range_train[t][i])
#     x_y_train.append([x_train, y_train])
#     lidar_range_train_list.append(lidar_range_train_sublist)

# df_test = pd.read_csv('data_test.csv', index_col=0)
lidar_range_test = df_test.values[:, 2:362]
# px_test = df_test["px"].values
# py_test = df_test["py"].values
# t_range_test = range(len(df_test))
# angle = np.linspace(-179, 180, num=360)
# x_y_test = []
# lidar_range_test_list = []
# for t in t_range_test:
#     x_test, y_test = [], []
#     lidar_range_test_sublist = []
#     for i in range(0, lidar_range_test.shape[1]):
#         if lidar_range_test[t][i] > 0:
#             x_test.append(px_test[t]+lidar_range_test[t][i]*np.cos(angle[i]/180*np.pi))
#             y_test.append(py_test[t]+lidar_range_test[t][i]*np.sin(angle[i]/180*np.pi))
#             lidar_range_test_sublist.append(lidar_range_test[t][i])
#     x_y_test.append([x_test, y_test])
#     lidar_range_test_list.append(lidar_range_test_sublist)

# Classify test dataset using train dataset
k_list = [1, 3, 5, 7, 9]
classes_train = df_train.values[:,-1]
classes_test = df_test.values[:, -1]
classifications_test = np.zeros(shape=(len(lidar_range_test), len(k_list)))
for k in range(len(k_list)):
    for i in range(len(lidar_range_test)):
        classifications_test[i, k] = KNNclassifier(k, lidar_range_test[i], lidar_range_test, classes_test)


classifications_train = np.zeros(shape=(len(lidar_range_test), len(k_list)))
for k in range(len(k_list)):
    for i in range(len(lidar_range_test)):
        classifications_train[i, k] = KNNclassifier(k, lidar_range_train[i], lidar_range_train, classes_train)

accuracy_test = dict()
for k in range(len(k_list)):
    accuracy_test[k_list[k]] = sum(classifications_test[:,k] == classes_test)/len(classes_test)

accuracy_train = dict()
for k in range(len(k_list)):
    accuracy_train[k_list[k]] = sum(classifications_train[:,k] == classes_train)/len(classes_train)

print(accuracy_test)
print(accuracy_train)
# # To complete

# import numpy as np
# np.random.seed(42)

# N_INPUTS = 10  #Number of inputs

# def mse_loss(y_true, y_pred):
#   return ((y_true - y_pred) ** 2).mean()

# # def sigmoid(x):
# # ...

# # def deriv_sigmoid(x):
# # ...

# # # ReLu activation function: 
# # def relu(x):
# # ...

# # # Derivative of ReLu
# # def deriv_relu(x):
# # ...

# class NeuralNetwork:
#   '''
#   Structure of the neural network:
#     - N_INPUTS inputs
#     - a hidden layer with 5 neurons (h1, h2, h3, h4, h5)
#     - an output layer with 1 neuron (o1)
#   '''
#   def __init__(self):
#     # Biases
#     # self.b1 = ...
#     # ...
#     # self.bo = np.random.random()

#     # # Weights
#     # self.w1o, self.w2o, self.w3o, self.w4o, self.w5o = np.random.random(5)
#     # self.wi1 = np.random.random(N_INPUTS)
#     # ...


#   def feedforward(self, x):
#     '''
#     - x is a numpy array with N_INPUTS elements.
#     '''
#     # # Hidden layer
#     # self.sum_h1 = np.dot(self.wi1, x) + self.b1
#     # ...
#     # self.h1 = relu(self.sum_h1)
#     # ...
 
#     # # Output layer
#     # self.sum_o1 = self.w1o*self.h1 + self.w2o*self.h2 + self.w3o*self.h3 + self.w4o*self.h4 + self.w5o*self.h5 + self.bo
#     # self.o1 = sigmoid(self.sum_o1)
#     # return self.o1

  
#   def train(self, data, y_trues, learn_rate = 0.1, epochs = 500):
#     '''
#     - data is a (n x N_INPUTS) numpy array, n = # of samples in the dataset.
#     - y_trues is a numpy array with n elements.
#       Elements in y_true correspond to those in data.
#     '''
#     loss_prev = 10000  #loss_prev is the loss of the previous iteration
#     for epoch in range(epochs):
#       for x, y_true in zip(data, y_trues):

#         # *************************************************
#         # 1. Feedforward Step
#         y_pred = self.feedforward(x)

#         # *************************************************
#         # 2. Backpropagation Step

#         # Partial derivatives.
#         d_L_d_ypred = -2 * (y_true - y_pred)

#         # Output Layer:  Neuron o1
#         # d_ypred_d_w1o = self.h1 * deriv_sigmoid(self.sum_o1)
#         # ...
#         # d_ypred_d_bo = deriv_sigmoid(self.sum_o1)

#         # d_ypred_d_h1 = self.w1o * deriv_sigmoid(self.sum_o1)
#         # ...

#         # Hidden Layer: Neuron h1
#         # d_h1_d_wi1 = ...
#         # d_h1_d_b1 = ...

#         # Hidden Layer: Neuron h1
#         # d_h2_d_wi2 = ...
#         # d_h2_d_b2 = ...

#         # Hidden Layer: Neuron h3

#         # Hidden Layer: Neuron h4

#         # Hidden Layer: Neuron h5

#         # *************************************************
#         # 3. Gradient Descent
#         # Output Layer:  Neuron o1
#         # self.w1o -= learn_rate * d_L_d_ypred * d_ypred_d_w1o
#         # ...
#         # self.bo -= learn_rate * d_L_d_ypred * d_ypred_d_bo

#         # Hidden Layer: Neuron h1
#         # self.wi1 -= ...
#         # self.b1 -= ...

#         # Hidden Layer: Neuron h2
#         # self.wi2 -= ...
#         # self.b2 -= ...

#         # Hidden Layer: Neuron h3

#         # Hidden Layer: Neuron h4

#         # Hidden Layer: Neuron h5

#       # *************************************************
#       # 4. Performance assessment (per epoch)
#       if epoch % 5 == 0:
#         y_preds = np.apply_along_axis(self.feedforward, 1, data)
#         loss = mse_loss(y_trues, y_preds)
#         print("Epoch %d  --> Loss: %.4f" % (epoch, loss))
# # Uncomment this part to enable a nonconstant learning rate 
# #         if loss > loss_prev:  #if loss did not decrease, let's decrease the learn_rate
# #           if learn_rate > 0.002:
# #             learn_rate = learn_rate*.9  #decrease 90% of the previous value
# #           print("I'm at epoch", epoch, "with new learn_rate: ", learn_rate)
# #         loss_prev = loss



# # Create the ANN
# model = NeuralNetwork()

# # # Build the Trainingset (with the first nonzero N_INPUTS ranges)
# # trainingset_X = np.zeros([len(Y_train), N_INPUTS])
# # for t in range(len(Y_train)):
# #   j=0
# #   for i in range(360):
# #     if X_train[t][i] > 0:
# #       if j < N_INPUTS:
# #         trainingset_X[t][j] = X_train[t][i]
# #         j +=1

# # # Trainingset: here the labels are 0 or 1
# # trainingset_Y = Y_train-1

# # #Shuffling the set...
# # from sklearn.utils import shuffle
# # trainingset_X, trainingset_Y = shuffle(trainingset_X, trainingset_Y, random_state=42)

# # # Train the ANN 
# # model.train(trainingset_X, trainingset_Y, learn_rate = 0.1, epochs = 1000)