import numpy as np 
import math


# train data
train = np.loadtxt('./bank-note/train.csv', delimiter =',',usecols = range(5))
#test data    
test = np.loadtxt('./bank-note/test.csv', delimiter =',',usecols = range(5))

# get vector x and y for both train and test datasets
X_train = train[:,:-1]
one_train = np.ones(X_train.shape[0])
D_train = np.column_stack((X_train,one_train))
Y_train = train[:,-1]
Y_train = 2 * Y_train - 1 

X_test = test[:,:-1]
one_test = np.ones(X_test.shape[0])
D_test = np.column_stack((X_test,one_test))
Y_test = test[:,-1]
Y_test = 2 * Y_test - 1

# set learning rate
r = 0.1
T = 10


def standard_alg(x, y, r, T):
	num_features = x.shape[1]
	num_samples = x.shape[0]
	w = np.zeros(num_features)
	idx = np.arange(num_samples)
	for t in range(T):
		np.random.shuffle(idx)
		x = x[idx,:]
		y = y[idx]
		for i in range(num_samples):
			temp = y[i] * np.dot(w, x[i])
			if temp <= 0:
				w = w+r*y[i]*x[i]

	return w 

def voted_alg(x, y, r, T):
	num_features = x.shape[1]
	num_samples = x.shape[0]
	w = np.zeros(num_features)
	idx = np.arange(num_samples)
	c_list = np.array([])
	w_list = np.array([])
	c = 0
	for t in range(T):
		np.random.shuffle(idx)
		x = x[idx,:]
		y = y[idx]
		for i in range(num_samples):
			temp = y[i] * np.dot(w, x[i])
			if temp <= 0:
				w_list = np.append(w_list, w)
				c_list = np.append(c_list, c)
				w = w+r*y[i]*x[i]
				c = 1 
			else:
				c +=1 

	l = c_list.shape[0]
	w_list = np.reshape(w_list, (l,-1))
	return c_list, w_list


def avg_alg(x, y, r, T):
	num_features = x.shape[1]
	num_samples = x.shape[0]
	w = np.zeros(num_features)
	a = np.zeros(num_features)
	idx = np.arange(num_samples)
	for t in range(T):
		np.random.shuffle(idx)
		x = x[idx,:]
		y = y[idx]
		for i in range(num_samples):
			temp = y[i] * np.dot(w, x[i])
			if temp <= 0:
				w = w+r*y[i]*x[i]
			a = a + w 

	return a 



# part a 
w = standard_alg(D_train, Y_train, r, T)
w = np.reshape(w, (-1,1))
pred = np.matmul(D_test, w)
pred[pred > 0] = 1
pred[pred <= 0] = -1
real = np.reshape(Y_test,(-1,1))
test_err = sum(abs(pred-real) / 2) / len(Y_test)


print("********** Part 2(a) **********")
print("Standard Perceptron:")
print()
print("learned weight vector: ", w)
print("Average prediction error: ", test_err)

# part b 
print()
print("********** Part 2(b) **********")
print("Voted Perceptron:")
print()
c_list, w_list = voted_alg(D_train, Y_train, r, T)
c_list = np.reshape(c_list, (-1,1))
print('weight vectors: ', w_list)
w_list = np.transpose(w_list)
prod = np.matmul(D_test, w_list)
prod[prod >0] = 1
prod[prod <=0] = -1
voted = np.matmul(prod, c_list)
voted[voted>0] = 1
voted[voted<=0] = -1 
real = np.reshape(Y_test,(-1,1)) 
test_err = sum(abs(voted-real) / 2) / len(Y_test)
print('Average test error: ', test_err)
print('count list: ', c_list)


# partc c 
print()
print("********** Part 2(c) **********")
print("Average Perceptron:")
print()

w = avg_alg(D_train, Y_train, r, T)
w = np.reshape(w, (-1,1))
pred = np.matmul(D_test, w)
pred[pred > 0] = 1
pred[pred <= 0] = -1
real = np.reshape(Y_test,(-1,1))
test_err = sum(abs(pred-real) / 2) / len(Y_test)
print("Learned weight vector: ", w)
print("Average test error: ", test_err)