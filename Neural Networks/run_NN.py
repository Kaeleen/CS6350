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
D_train = np.column_stack((one_train,X_train))
Y_train = train[:,-1]
Y_train = 2 * Y_train - 1 
D_train = np.matrix(D_train)
Y_train = np.array(Y_train)

X_test = test[:,:-1]
one_test = np.ones(X_test.shape[0])
D_test = np.column_stack((one_test,X_test))
Y_test = test[:,-1]
Y_test = 2 * Y_test - 1
D_test = np.matrix(D_test)
Y_test = np.array(Y_test)

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

print()
print("********** Part 3(b) **********")


T = 100

gamma_list = [                 # gamma0, d
	NN.GammaSchedule(1/8720, 40), # 0.15, 30 - 0.1, 35
	NN.GammaSchedule(1/17440, 25), # 10 0.1, 15 - 0.05, 20
	NN.GammaSchedule(1/34880, 35), # 25 0.1, 20 - 0.05, 25 - 0.05, 30
	NN.GammaSchedule(7/87200, 25), # 50 0.075, 17.5 - 0.07, 18
	NN.GammaSchedule(1/87200, 10)  # 100 0.02, 2 - 0.01, 2.5
	]

width_list = [5, 10, 25, 50, 100]


print("Width\tTrain Error\tTest Error")
for i, width in enumerate(width_list):
	nn = NN.NeuralNetwork(3, D_train.shape[1], [width, width], True)
	
	nn_learned = NN.sgd(D_train, Y_train, nn, gamma_list[i], T)

	train_predicts = NN.sgd_predict(D_train, nn_learned)
	numWrong = sum(abs(train_predicts-Y_train) / 2)
	train_err = numWrong/len(Y_train)

	test_predicts = NN.sgd_predict(D_test, nn_learned)
	numWrong = sum(abs(test_predicts-Y_test) / 2)
	test_err = numWrong/len(Y_test)

	print(f"{width}\t{train_err:.8f}\t{test_err:.8f}")


print()
print("********** Part 3c **********")

print("Width\tTrain Error\tTest Error")
for i, width in enumerate(width_list):
    nn = NN.NeuralNetwork(3, D_train.shape[1], [width, width], False)
    nn_learned = NN.sgd(D_train, Y_train, nn, gamma_list[i], T)

    train_predicts = NN.sgd_predict(D_train, nn_learned)
    numWrong = sum(abs(train_predicts-Y_train) / 2)
    train_err = numWrong/len(Y_train)

    test_predicts = NN.sgd_predict(D_test, nn_learned)
    numWrong = sum(abs(test_predicts-Y_test) / 2)
    test_err = numWrong/len(Y_test)

    print(f"{width}\t{train_err:.8f}\t{test_err:.8f}")