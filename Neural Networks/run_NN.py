import numpy as np 
import math
import NN
import pandas as pd 

# train data
train = np.loadtxt('./bank-note/train.csv', delimiter =',',usecols = range(5))
#test data    
test = np.loadtxt('./bank-note/test.csv', delimiter =',',usecols = range(5))

# get vector x and y for both train and test datasets
X_train = train[:,:-1]
one_train = np.ones(X_train.shape[0])
D_train = np.column_stack((X_train,one_train))
Y_train = train[:,-1]
# Y_train = 2 * Y_train - 1 

X_test = test[:,:-1]
one_test = np.ones(X_test.shape[0])
D_test = np.column_stack((X_test,one_test))
Y_test = test[:,-1]
# Y_test = 2 * Y_test - 1

print()
print("********** Part 3(a) **********")
print()
print("back propagation test:")

nn = NN.NeuralNetwork(3, 3, [2, 2], False)

# Given weights
w = [
        [ 
            [ 0,  0,  0], 
            [-1, -2, -3], 
            [ 1,  2,  3]  
        ],
        [ 
            [ 0,  0,  0], 
            [-1, -2, -3], 
            [ 1,  2,  3]
        ],
        [
            [0, 0, 0],
            [-1, 2, -1.5],
            [0, 0, 0]
        ]
    ]
nn.weights = np.array(w)

n = [
        [1, 1, 1], # input
        [1, 0.00247, 0.9975], # hidden layer 1
        [1, 0.01803, 0.98197] # hidden layer 2
    ]
nn.nodes = np.array(n)

nn.y = -2.4369

NN.backward(1, nn)
print(f"dw: {nn.dweights}")
print()
print("forward pass test:")

NN.forward(np.matrix([1,1,1]), nn)
print(f"nodes: {nn.nodes}")
print(f"y: {nn.y}")