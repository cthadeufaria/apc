import pandas as pd
import numpy as np

df_train = pd.read_csv('project02/data_train.csv', index_col=0)
df_test = pd.read_csv('project02/data_test.csv', index_col=0)

# df_train = pd.read_csv('data_train.csv', index_col=0)
lidar_range_train = df_train.values[:, 2:362]
px_train = df_train["px"].values
py_train = df_train["py"].values
t_range_train = range(len(df_train))
angle = np.linspace(-179, 180, num=360)
x_y_train = []
lidar_range_train_list = []
for t in t_range_train:
    x_train, y_train = [], []
    lidar_range_train_sublist = []
    for i in range(0, lidar_range_train.shape[1]):
        if lidar_range_train[t][i] > 0:
            x_train.append(px_train[t]+lidar_range_train[t][i]*np.cos(angle[i]/180*np.pi))
            y_train.append(py_train[t]+lidar_range_train[t][i]*np.sin(angle[i]/180*np.pi))
            lidar_range_train_sublist.append(lidar_range_train[t][i])
    x_y_train.append([x_train, y_train])
    lidar_range_train_list.append(lidar_range_train_sublist)

classes_train = df_train.values[:,-1]

import numpy as np

def mse_loss(y_true, y_pred):
    return((y_true - y_pred)**2).mean()

def relu(x):
    return np.maximum(x, 0)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu_derivative(x):
    if x > 0:
        return 1.
    elif x <= 0:
        return 0.

def sigmoid_derivative(x):
    fx = sigmoid(x)
    return fx * (1 - fx)

class NeuralNetwork:
    def __init__(self) -> None:
        # biases
        self.biases = np.zeros(shape=(6,))
        for i in range(self.biases.shape[0]):
            self.biases[i] = np.random.random()

        # weights
        self.weights = np.zeros(shape=(55,))
        for i in range(self.weights.shape[0]):
            self.weights[i] = np.random.random()

    def feedforward(self, x_list):
        x = np.array(x_list)
        self.total_h1 = np.dot(x, self.weights[0:10]) + self.biases[0]
        self.total_h2 = np.dot(x, self.weights[10:20]) + self.biases[1]
        self.total_h3 = np.dot(x, self.weights[20:30]) + self.biases[2]
        self.total_h4 = np.dot(x, self.weights[30:40]) + self.biases[3]
        self.total_h5 = np.dot(x, self.weights[40:50]) + self.biases[4]
        self.output_h1 = relu(self.total_h1)
        self.output_h2 = relu(self.total_h2)
        self.output_h3 = relu(self.total_h3)
        self.output_h4 = relu(self.total_h4)
        self.output_h5 = relu(self.total_h5)

        self.total_o1 = np.dot(
            np.array([self.output_h1, self.output_h2, self.output_h3, self.output_h4, self.output_h5]),
            self.weights[50:55]
        ) + self.biases[5]
        self.output_o1 = sigmoid(self.total_o1)

        return self.output_o1
    
    def train(self, input, y_trues, learning_rate=0.1, epochs=300):
        for epoch in range(epochs):
            for x, y_true in zip(input, y_trues):
                y_pred = self.feedforward(x)

                # Backpropagation ###########################

                dl_dypred = -2*(y_true - y_pred)
                print(y_true,  y_pred)

                # output layer / neuron o1
                dypred_dw51 = self.output_h1 * sigmoid_derivative(self.total_o1)
                dypred_dw52 = self.output_h2 * sigmoid_derivative(self.total_o1)
                dypred_dw53 = self.output_h3 * sigmoid_derivative(self.total_o1)
                dypred_dw54 = self.output_h4 * sigmoid_derivative(self.total_o1)
                dypred_dw55 = self.output_h5 * sigmoid_derivative(self.total_o1)
                dypred_db5 = sigmoid_derivative(self.total_o1)
                dypred_dh1 = self.weights[50] * sigmoid_derivative(self.total_o1)
                dypred_dh2 = self.weights[51] * sigmoid_derivative(self.total_o1)
                dypred_dh3 = self.weights[52] * sigmoid_derivative(self.total_o1)
                dypred_dh4 = self.weights[53] * sigmoid_derivative(self.total_o1)
                dypred_dh5 = self.weights[54] * sigmoid_derivative(self.total_o1)

                # hidden layer / neuron h1
                dh1_db1 = relu_derivative(self.total_h1)
                dh1_dw1 = x[0] * relu_derivative(self.total_h1)
                dh1_dw2 = x[1] * relu_derivative(self.total_h1)
                dh1_dw3 = x[2] * relu_derivative(self.total_h1)
                dh1_dw4 = x[3] * relu_derivative(self.total_h1)
                dh1_dw5 = x[4] * relu_derivative(self.total_h1)
                dh1_dw6 = x[5] * relu_derivative(self.total_h1)
                dh1_dw7 = x[6] * relu_derivative(self.total_h1)
                dh1_dw8 = x[7] * relu_derivative(self.total_h1)
                dh1_dw9 = x[8] * relu_derivative(self.total_h1)
                dh1_dw10 = x[9] * relu_derivative(self.total_h1)

                # hidden layer / neuron h2
                dh2_db2 = relu_derivative(self.total_h2)
                dh2_dw11 = x[0] * relu_derivative(self.total_h2)
                dh2_dw12 = x[1] * relu_derivative(self.total_h2)
                dh2_dw13 = x[2] * relu_derivative(self.total_h2)
                dh2_dw14 = x[3] * relu_derivative(self.total_h2)
                dh2_dw15 = x[4] * relu_derivative(self.total_h2)
                dh2_dw16 = x[5] * relu_derivative(self.total_h2)
                dh2_dw17 = x[6] * relu_derivative(self.total_h2)
                dh2_dw18 = x[7] * relu_derivative(self.total_h2)
                dh2_dw19 = x[8] * relu_derivative(self.total_h2)
                dh2_dw20 = x[9] * relu_derivative(self.total_h2)

                # hidden layer / neuron h3
                dh3_db3 = relu_derivative(self.total_h3)
                dh3_dw21 = x[0] * relu_derivative(self.total_h3)
                dh3_dw22 = x[1] * relu_derivative(self.total_h3)
                dh3_dw23 = x[2] * relu_derivative(self.total_h3)
                dh3_dw24 = x[3] * relu_derivative(self.total_h3)
                dh3_dw25 = x[4] * relu_derivative(self.total_h3)
                dh3_dw26 = x[5] * relu_derivative(self.total_h3)
                dh3_dw27 = x[6] * relu_derivative(self.total_h3)
                dh3_dw28 = x[7] * relu_derivative(self.total_h3)
                dh3_dw29 = x[8] * relu_derivative(self.total_h3)
                dh3_dw30 = x[9] * relu_derivative(self.total_h3)

                # hidden layer / neuron h4
                dh4_db4 = relu_derivative(self.total_h4)
                dh4_dw31 = x[0] * relu_derivative(self.total_h4)
                dh4_dw32 = x[1] * relu_derivative(self.total_h4)
                dh4_dw33 = x[2] * relu_derivative(self.total_h4)
                dh4_dw34 = x[3] * relu_derivative(self.total_h4)
                dh4_dw35 = x[4] * relu_derivative(self.total_h4)
                dh4_dw36 = x[5] * relu_derivative(self.total_h4)
                dh4_dw37 = x[6] * relu_derivative(self.total_h4)
                dh4_dw38 = x[7] * relu_derivative(self.total_h4)
                dh4_dw39 = x[8] * relu_derivative(self.total_h4)
                dh4_dw40 = x[9] * relu_derivative(self.total_h4)

                # hidden layer / neuron h5
                dh5_db5 = relu_derivative(self.total_h5)
                dh5_dw41 = x[0] * relu_derivative(self.total_h5)
                dh5_dw42 = x[1] * relu_derivative(self.total_h5)
                dh5_dw43 = x[2] * relu_derivative(self.total_h5)
                dh5_dw44 = x[3] * relu_derivative(self.total_h5)
                dh5_dw45 = x[4] * relu_derivative(self.total_h5)
                dh5_dw46 = x[5] * relu_derivative(self.total_h5)
                dh5_dw47 = x[6] * relu_derivative(self.total_h5)
                dh5_dw48 = x[7] * relu_derivative(self.total_h5)
                dh5_dw49 = x[8] * relu_derivative(self.total_h5)
                dh5_dw50 = x[9] * relu_derivative(self.total_h5)

                #######################################################

                # Gradient Descent
                # Output layer / neuron o1
                if y_true == 2:
                    pass
                self.weights[50] -= learning_rate * dl_dypred * dypred_dw51
                self.weights[51] -= learning_rate * dl_dypred * dypred_dw52
                self.weights[52] -= learning_rate * dl_dypred * dypred_dw53
                self.weights[53] -= learning_rate * dl_dypred * dypred_dw54
                self.weights[54] -= learning_rate * dl_dypred * dypred_dw55
                self.biases[4] -= learning_rate * dl_dypred * dypred_db5

                # Hidden layer / neuron h1
                self.weights[0] -= learning_rate * dl_dypred * dypred_dh1 * dh1_dw1
                self.weights[1] -= learning_rate * dl_dypred * dypred_dh1 * dh1_dw2
                self.weights[2] -= learning_rate * dl_dypred * dypred_dh1 * dh1_dw3
                self.weights[3] -= learning_rate * dl_dypred * dypred_dh1 * dh1_dw4
                self.weights[4] -= learning_rate * dl_dypred * dypred_dh1 * dh1_dw5
                self.weights[5] -= learning_rate * dl_dypred * dypred_dh1 * dh1_dw6
                self.weights[6] -= learning_rate * dl_dypred * dypred_dh1 * dh1_dw7
                self.weights[7] -= learning_rate * dl_dypred * dypred_dh1 * dh1_dw8
                self.weights[8] -= learning_rate * dl_dypred * dypred_dh1 * dh1_dw9
                self.weights[9] -= learning_rate * dl_dypred * dypred_dh1 * dh1_dw10
                self.biases[0] -= learning_rate * dl_dypred * dypred_dh1 * dh1_db1

                # Hidden layer / neuron h2
                self.weights[10] -= learning_rate * dl_dypred * dypred_dh2 * dh2_dw11
                self.weights[11] -= learning_rate * dl_dypred * dypred_dh2 * dh2_dw12
                self.weights[12] -= learning_rate * dl_dypred * dypred_dh2 * dh2_dw13
                self.weights[13] -= learning_rate * dl_dypred * dypred_dh2 * dh2_dw14
                self.weights[14] -= learning_rate * dl_dypred * dypred_dh2 * dh2_dw15
                self.weights[15] -= learning_rate * dl_dypred * dypred_dh2 * dh2_dw16
                self.weights[16] -= learning_rate * dl_dypred * dypred_dh2 * dh2_dw17
                self.weights[17] -= learning_rate * dl_dypred * dypred_dh2 * dh2_dw18
                self.weights[18] -= learning_rate * dl_dypred * dypred_dh2 * dh2_dw19
                self.weights[19] -= learning_rate * dl_dypred * dypred_dh2 * dh2_dw20
                self.biases[1] -= learning_rate * dl_dypred * dypred_dh2 * dh2_db2

                # Hidden layer / neuron h3
                self.weights[20] -= learning_rate * dl_dypred * dypred_dh3 * dh3_dw21
                self.weights[21] -= learning_rate * dl_dypred * dypred_dh3 * dh3_dw22
                self.weights[22] -= learning_rate * dl_dypred * dypred_dh3 * dh3_dw23
                self.weights[23] -= learning_rate * dl_dypred * dypred_dh3 * dh3_dw24
                self.weights[24] -= learning_rate * dl_dypred * dypred_dh3 * dh3_dw25
                self.weights[25] -= learning_rate * dl_dypred * dypred_dh3 * dh3_dw26
                self.weights[26] -= learning_rate * dl_dypred * dypred_dh3 * dh3_dw27
                self.weights[27] -= learning_rate * dl_dypred * dypred_dh3 * dh3_dw28
                self.weights[28] -= learning_rate * dl_dypred * dypred_dh3 * dh3_dw29
                self.weights[29] -= learning_rate * dl_dypred * dypred_dh3 * dh3_dw30
                self.biases[2] -= learning_rate * dl_dypred * dypred_dh3 * dh3_db3
            
                # Hidden layer / neuron h4
                self.weights[30] -= learning_rate * dl_dypred * dypred_dh4 * dh4_dw31
                self.weights[31] -= learning_rate * dl_dypred * dypred_dh4 * dh4_dw32
                self.weights[32] -= learning_rate * dl_dypred * dypred_dh4 * dh4_dw33
                self.weights[33] -= learning_rate * dl_dypred * dypred_dh4 * dh4_dw34
                self.weights[34] -= learning_rate * dl_dypred * dypred_dh4 * dh4_dw35
                self.weights[35] -= learning_rate * dl_dypred * dypred_dh4 * dh4_dw36
                self.weights[36] -= learning_rate * dl_dypred * dypred_dh4 * dh4_dw37
                self.weights[37] -= learning_rate * dl_dypred * dypred_dh4 * dh4_dw38
                self.weights[38] -= learning_rate * dl_dypred * dypred_dh4 * dh4_dw39
                self.weights[39] -= learning_rate * dl_dypred * dypred_dh4 * dh4_dw40
                self.biases[3] -= learning_rate * dl_dypred * dypred_dh4 * dh4_db4

                # Hidden layer / neuron h5
                self.weights[40] -= learning_rate * dl_dypred * dypred_dh5 * dh5_dw41
                self.weights[41] -= learning_rate * dl_dypred * dypred_dh5 * dh5_dw42
                self.weights[42] -= learning_rate * dl_dypred * dypred_dh5 * dh5_dw43
                self.weights[43] -= learning_rate * dl_dypred * dypred_dh5 * dh5_dw44
                self.weights[44] -= learning_rate * dl_dypred * dypred_dh5 * dh5_dw45
                self.weights[45] -= learning_rate * dl_dypred * dypred_dh5 * dh5_dw46
                self.weights[46] -= learning_rate * dl_dypred * dypred_dh5 * dh5_dw47
                self.weights[47] -= learning_rate * dl_dypred * dypred_dh5 * dh5_dw48
                self.weights[48] -= learning_rate * dl_dypred * dypred_dh5 * dh5_dw49
                self.weights[49] -= learning_rate * dl_dypred * dypred_dh5 * dh5_dw50
                self.biases[4] -= learning_rate * dl_dypred * dypred_dh5 * dh5_db5

            # Performance assessment
            y_preds = np.apply_along_axis(self.feedforward, 1, input)
            loss = mse_loss(y_trues, y_preds)
            print("Epoch %d | Loss: %.4f" % (epoch, loss))

# x_y_train_10 = [[x[0:10], y[0:10]] for x, y in x_y_train]
lidar_range_train_list_10 = [x[0:10] for x in lidar_range_train_list]
model = NeuralNetwork()
model.train(lidar_range_train_list_10, classes_train)