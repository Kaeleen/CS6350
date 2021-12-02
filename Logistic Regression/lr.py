import numpy as np 
import warnings

warnings.filterwarnings('ignore')
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


lr = 0.01
d = 0.1
T = 100

v_list = [0.01, 0.1, 0.5, 1, 3, 5, 10, 100]

def train_MAP( x, y, v, lr):
	num_sample = x.shape[0]
	dim = x.shape[1]
	w = np.zeros([1, dim])
	idx = np.arange(num_sample)
	for t in range(T):
		np.random.shuffle(idx)
		x = x[idx,:]
		y = y[idx]
		for i in range(num_sample):
			x_i = x[i,:].reshape([1, -1])
			tmp = y[i] * np.sum(np.multiply(w, x_i))
			g = - num_sample * y[i] * x_i / (1 + np.exp(tmp)) + w / v

			lr = lr / (1 + lr / d * t)
			w = w - lr * g
	return w.reshape([-1,1])


def train_ML(x, y, lr):
	num_sample = x.shape[0]
	dim = x.shape[1]
	w = np.zeros([1, dim])
	idx = np.arange(num_sample)
	for t in range(T):
		np.random.shuffle(idx)
		x = x[idx,:]
		y = y[idx]
		for i in range(num_sample):
			tmp = y[i] * np.sum(np.multiply(w, x[i,:]))
			g = - num_sample * y[i] * x[i,:] / (1 + np.exp(tmp))
			lr = lr / (1 + lr / d * t)
			w = w - lr * g
	return w.reshape([-1,1])

print("********** Part 3(a) **********")
print("variance\tTrain Error\tTest Error")
print()
for v in v_list:

	w= train_MAP(D_train, Y_train, v, lr)


	pred = np.matmul(D_train, w)
	pred[pred > 0] = 1
	pred[pred <= 0] = -1
	train_err = np.sum(np.abs(pred - np.reshape(Y_train,(-1,1)))) / 2 / Y_train.shape[0]

	pred = np.matmul(D_test, w)
	pred[pred > 0] = 1
	pred[pred <= 0] = -1

	test_err = np.sum(np.abs(pred - np.reshape(Y_test,(-1,1)))) / 2 / Y_test.shape[0]
	print(f"{v}\t\t{train_err:.8f}\t{test_err:.8f}")

print()
print("********** Part 3(b) **********")
print("variance\tTrain Error\tTest Error")
print()
for v in v_list:

	w= train_ML(D_train, Y_train, lr)


	pred = np.matmul(D_train, w)
	pred[pred > 0] = 1
	pred[pred <= 0] = -1
	train_err = np.sum(np.abs(pred - np.reshape(Y_train,(-1,1)))) / 2 / Y_train.shape[0]

	pred = np.matmul(D_test, w)
	pred[pred > 0] = 1
	pred[pred <= 0] = -1

	test_err = np.sum(np.abs(pred - np.reshape(Y_test,(-1,1)))) / 2 / Y_test.shape[0]
	print(f"{v}\t\t{train_err:.8f}\t{test_err:.8f}")
