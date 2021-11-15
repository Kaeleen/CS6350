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

lr = 0.1 
a = 0.1 
T = 100


C_set = np.array([float(100/873), float(500/873), float(700/873)])

def two_a(x, y, C, lr=0.1):
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
			g = np.copy(w)
			g[num_features-1] = 0
			if temp <= 1:
					g = g - C * num_samples * y[i] * x[i,:]
			lr = lr / (1 + lr / a * t)
			w = w - lr * g
	return w


def two_b(x, y, C, lr=0.1):
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
			g = np.copy(w)
			g[num_features-1] = 0
			if temp <= 1:
					g = g - C * num_samples * y[i] * x[i,:]
			lr = lr / (1 + t)
			w = w - lr * g
	return w


print("********** Part 2(a) **********")
for C in C_set:
	w = two_a(D_train, Y_train, C, lr)
	w = np.reshape(w, (5,1))

	pred = np.matmul(D_train, w)
	pred[pred > 0] = 1
	pred[pred <= 0] = -1
	train_err = np.sum(np.abs(pred - np.reshape(Y_train,(-1,1)))) / 2 / Y_train.shape[0]

	pred = np.matmul(D_test, w)
	pred[pred > 0] = 1
	pred[pred <= 0] = -1

	test_err = np.sum(np.abs(pred - np.reshape(Y_test,(-1,1)))) / 2 / Y_test.shape[0]
	print('linear SVM Primal train_error: ', train_err, ' test_error: ', test_err)
	w = np.reshape(w, (1,-1))

print()
print("********** Part 2(b) **********")
for C in C_set:
	w = two_b(D_train, Y_train, C, lr)
	w = np.reshape(w, (5,1))

	pred = np.matmul(D_train, w)
	pred[pred > 0] = 1
	pred[pred <= 0] = -1
	train_err = np.sum(np.abs(pred - np.reshape(Y_train,(-1,1)))) / 2 / Y_train.shape[0]

	pred = np.matmul(D_test, w)
	pred[pred > 0] = 1
	pred[pred <= 0] = -1

	test_err = np.sum(np.abs(pred - np.reshape(Y_test,(-1,1)))) / 2 / Y_test.shape[0]
	print('linear SVM Primal train_error: ', train_err, ' test_error: ', test_err)
	w = np.reshape(w, (1,-1))