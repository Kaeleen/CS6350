import numpy as np 
import math
import scipy.optimize as opt
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
gamma_set = np.array([0.1, 0.5, 1, 5, 100])

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

def con(alpha,y):
	t = np.matmul(np.reshape(alpha,(1, -1)), np.reshape(y,(-1,1)))
	return t[0]


def obj(alpha, x, y):
	l = 0
	l = l - np.sum(alpha)
	ayx = np.multiply(np.multiply(np.reshape(alpha,(-1,1)), np.reshape(y, (-1,1))), x)
	l = l + 0.5 * np.sum(np.matmul(ayx, np.transpose(ayx)))
	return l


def dual(x, y, C):
	num_samples = x.shape[0]
	bnds = [(0, C)] * num_samples
	cons = ({'type': 'eq', 'fun': lambda alpha: con(alpha, y)})
	alpha0 = np.zeros(num_samples)
	res = opt.minimize(lambda alpha: obj(alpha, x, y), alpha0,  method='SLSQP', bounds=bnds,constraints=cons, options={'disp': False})
	
	w = np.sum(np.multiply(np.multiply(np.reshape(res.x,(-1,1)), np.reshape(y, (-1,1))), x), axis=0)
	idx = np.where((res.x > 0) & (res.x < C))
	b =  np.mean(y[idx] - np.matmul(x[idx,:], np.reshape(w, (-1,1))))
	w = w.tolist()
	w.append(b)
	w = np.array(w)
	return w


def gaussian_kernel(x1, x2, gamma):
	m1 = np.tile(x1, (1, x2.shape[0]))
	m1 = np.reshape(m1, (-1,x1.shape[1]))
	m2 = np.tile(x2, (x1.shape[0], 1))
	k = np.exp(np.sum(np.square(m1 - m2),axis=1) / -gamma)
	k = np.reshape(k, (x1.shape[0], x2.shape[0]))
	return k

def obj_gk(alpha, k, y):
	l = 0
	l = l - np.sum(alpha)
	ay = np.multiply(np.reshape(alpha,(-1,1)), np.reshape(y, (-1,1)))
	ayay = np.matmul(ay, np.transpose(ay))
	l = l + 0.5 * np.sum(np.multiply(ayay, k))
	return l


def train_gaussian_kernel(x, y, C, gamma):
	num_samples = x.shape[0]
	bnds = [(0, C)] * num_samples
	cons = ({'type': 'eq', 'fun': lambda alpha: con(alpha, y)})
	alpha0 = np.zeros(num_samples)
	k = gaussian_kernel(x, x, gamma)
	res = opt.minimize(lambda alpha: obj_gk(alpha, k, y), alpha0,  method='SLSQP', bounds=bnds,constraints=cons, options={'disp': False})
	return res.x


def predict_gaussian_kernel(alpha, x0, y0, x, gamma):
	k = gaussian_kernel(x0, x, gamma)
	k = np.multiply(np.reshape(y0, (-1,1)), k)
	y = np.sum(np.multiply(np.reshape(alpha, (-1,1)), k), axis=0)
	y = np.reshape(y, (-1,1))
	y[y > 0] = 1
	y[y <=0] = -1
	return y


def Perceptron_Kernel_Gaussian(x, y, g, T):
	c = np.zeros(x.shape[0])
	idxs = np.arange(x.shape[0])

	k = gaussian_kernel(x, x, g)
	
	for epoch in range(T):
	 
		np.random.shuffle(idxs)

		
		for i in idxs:
			p = np.sum(c * y * k[:,i]) 
			sgn = 1 if p > 0 else -1
			if sgn != y[i]:
				c[i] += 1
	return c


def predict_Perceptron_Kernel_Gaussian(x, c, x_train, y, g):
	predictions = []
	for ex in x:
		p = 0
		for i in range(c.shape[0]):
			k = math.exp(-1 * np.linalg.norm(x_train[i] - ex)**2 / g)
			p += np.asscalar(c[i] * y[i] * k)
		if p < 0:
			predictions.append(-1)
		else:
			predictions.append(1)
	return predictions



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
	print('train_error: ', train_err, ' test_error: ', test_err)
	w = np.reshape(w, (1,-1))
	print("learnt weights:", w)

print()
print("********** Part 2(b) **********")
for C in C_set:
	w1 = two_b(D_train, Y_train, C, lr)
	w1 = np.reshape(w1, (5,1))

	pred = np.matmul(D_train, w1)
	pred[pred > 0] = 1
	pred[pred <= 0] = -1
	train_err = np.sum(np.abs(pred - np.reshape(Y_train,(-1,1)))) / 2 / Y_train.shape[0]

	pred = np.matmul(D_test, w1)
	pred[pred > 0] = 1
	pred[pred <= 0] = -1

	test_err = np.sum(np.abs(pred - np.reshape(Y_test,(-1,1)))) / 2 / Y_test.shape[0]
	print('train_error: ', train_err, ' test_error: ', test_err)
	w1 = np.reshape(w1, (1,-1))
	print("learnt weights:",w1)


print()
print("********** Part 3(a) **********")
for C in C_set:
	w2 = dual(D_train[:,[x for x in range(4)]] ,Y_train, C)

	w2 = np.reshape(w2, (5,1))

	pred = np.matmul(D_train, w2)
	pred[pred > 0] = 1
	pred[pred <= 0] = -1
	train_err = np.sum(np.abs(pred - np.reshape(Y_train,(-1,1)))) / 2 / Y_train.shape[0]

	pred = np.matmul(D_test, w2)
	pred[pred > 0] = 1
	pred[pred <= 0] = -1

	test_err = np.sum(np.abs(pred - np.reshape(Y_test,(-1,1)))) / 2 / Y_test.shape[0]
	print('linear SVM Dual train_error: ', train_err, ' test_error: ', test_err)

print()
print("********** Part 3(b) and 3(c)**********")
for C in C_set:
	c = 0
	for gamma in gamma_set:
		print('gamma: ', gamma, 'C:', C)
		alpha = train_gaussian_kernel(D_train[:,[x for x in range(4)]] ,Y_train, C, gamma)
		idx = np.where(alpha > 0)[0]
		print('# of sv: ', len(idx))
		# train 
		y = predict_gaussian_kernel(alpha, D_train[:,[x for x in range(4)]], Y_train, D_train[:,[x for x in range(4)]], gamma)
		train_err = np.sum(np.abs(y - np.reshape(Y_train,(-1,1)))) / 2 / Y_train.shape[0]

		# test
		y = predict_gaussian_kernel(alpha, D_train[:,[x for x in range(4)]], Y_train, X_test[:,[x for x in range(4)]], gamma)
		test_err = np.sum(np.abs(y - np.reshape(Y_test,(-1,1)))) / 2 / Y_test.shape[0]
		print('train_error: ', train_err, ' test_error: ', test_err)
		
		if c > 0:
			intersect = len(np.intersect1d(idx, old_idx))
			print('# of overlapped: ', intersect)
		c = c + 1
		old_idx = idx


print()
print("********** Part 3(d)**********")


for g in gamma_set:
	c = Perceptron_Kernel_Gaussian(X_train, Y_train, g, T)
	predict_train = predict_Perceptron_Kernel_Gaussian(X_train, c, X_train, Y_train, g)
	missed = sum(abs(predict_train-Y_train) / 2)
	train_err = missed/len(Y_train)

	predict_test = predict_Perceptron_Kernel_Gaussian(X_test, c, X_train, Y_train, g)
	missed = sum(abs(predict_test-Y_test) / 2)
	test_err = missed/len(Y_test)
	
	print('gamma: ', g, 'train_error: ', train_err, ' test_error: ', test_err)
