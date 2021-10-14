import numpy as np 
import math
import batchGradientDescent as BGD
import stochasticGradientDescent as SGD
import matplotlib.pyplot as plt
from numpy.linalg import inv

# train data
train = np.loadtxt('./concrete/train.csv', delimiter =',',usecols = range(8))
#test data    
test = np.loadtxt('./concrete/test.csv', delimiter =',',usecols = range(8))

# get vector x and y for both train and test datasets
X_train = train[:,:-1]
one_train = np.ones(X_train.shape[0])
D_train = np.column_stack((one_train, X_train))
Y_train = train[:,-1]

X_test = test[:,:-1]
one_test = np.ones(X_test.shape[0])
D_test = np.column_stack((one_test, X_test))
Y_test = test[:,-1]

print()
# part a 
print("********** Part 4(a) **********")
print("Batched gradient descent experiment")

r = 0.01
W, costs = BGD.LMS_gradiant(D_train, Y_train, r)
test_cost_value = BGD.cost_function(D_test, Y_test, W)
print("Learning rate: ", r)
print("The learned weight vector: ", W)
print("Test data cost function value: ", test_cost_value)
fig1 = plt.figure()
plt.plot(costs)
fig1.suptitle('Gradient Descent ', fontsize=20)
plt.xlabel('iteration', fontsize=18)
plt.ylabel('Cost Function Value', fontsize=16)
plt.show()
fig1.savefig("BGD_cost_function.png")
print("Figure has been saved!")

# part b
print()
print()
print("********** Part 4(b) **********")
print("stochastic gradient descent experiment")

r = 0.001
W, costs = SGD.sgd(D_train, Y_train, r)
test_cost_value = SGD.cost_function(D_test, Y_test, W)
print("Learning rate: ", r)
print("The learned weight vector: ", W)
print("Test data cost function value: ", test_cost_value)
fig2 = plt.figure()
plt.plot(costs)
fig2.suptitle('Stochastic Gradient Descent ', fontsize=20)
plt.xlabel('iteration', fontsize=18)
plt.ylabel('Cost Function Value', fontsize=16)
plt.show()
fig2.savefig("SGD_cost_function.png")
print("The figure has been saved! ")

# part c
print()
print()
print("********** Part 4(c) **********")
print("Find optimal weight vector with analytical form")

new_D_train = D_train.T
temp = np.matmul(new_D_train, new_D_train.T)
invtemp = inv(temp)
final_w = np.matmul(np.matmul(invtemp, new_D_train), Y_train)
test_cost_value = SGD.cost_function(D_test, Y_test, final_w)
print("The learned weight vector: ", final_w)
print("Test data cost function value: ", test_cost_value)