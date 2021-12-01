import torch
from torch import nn
from torch.nn import Module, ReLU, ModuleList, Parameter, Tanh
import numpy as np

class SelfLinear(Module):
	def __init__(self, n_in, n_out):
		super(SelfLinear, self).__init__()
		# Use Xavier initialization
		w = torch.empty(n_out, n_in)
		nn.init.xavier_uniform_(w)
		self.weight = Parameter(w.double())
		b = torch.empty(1, n_out)
		nn.init.xavier_uniform_(b)
		self.bias = Parameter(b.double())
	
	def forward(self, X):
		return X @ self.weight.T  + self.bias
	
class Net(Module):
	def __init__(self, layers):
		super(Net, self).__init__()
		self.act = Tanh()
		self.fcs = ModuleList()
		self.layers = layers
		
		for i in range(len(self.layers)-1):
			self.fcs.append(SelfLinear(self.layers[i], self.layers[i+1]))
			
	def forward(self, X):
		for fc in self.fcs[:-1]:
			X = fc(X)
			X = self.act(X)
		X = self.fcs[-1](X)
		return X



# train data
train = np.loadtxt('./bank-note/train.csv', delimiter =',',usecols = range(5))
#test data    
test = np.loadtxt('./bank-note/test.csv', delimiter =',',usecols = range(5))

X_train = train[:,:-1]
one_train = np.ones(X_train.shape[0])
D_train = np.column_stack((X_train,one_train))
Y_train = np.matrix(train[:,-1]).T
Y_train = 2 * Y_train - 1 


X_test = test[:,:-1]
one_test = np.ones(X_test.shape[0])
D_test = np.column_stack((X_test,one_test))
Y_test = np.matrix(test[:,-1]).T
Y_test = 2 * Y_test - 1


# Convert the data to PyTorch tensors
x = torch.tensor(D_train)
y = torch.tensor(Y_train)

x_test_tensor = torch.tensor(D_test)
y_test_tensor = torch.tensor(Y_test)


depth = [3, 5, 9]
width = [5, 10, 25, 50, 100]

print()
print("PyTorch using RELU activation function")

print("Depth\tWidth\tTrain Error\tTest Error")
for d in depth:
	for w in width:
		layers = [x.shape[1]]
		layers += ([w for i in range(d)])
		layers += [1]
		model = Net(layers)
		optimizer = torch.optim.Adam(model.parameters())

		
		for epoch in range(100):
			optimizer.zero_grad()
			L = ((model(x) - y)**2).sum()
			L.backward()
			optimizer.step()

		
		with torch.no_grad():
			y_train_pred = (model(x)).detach().numpy()
			y_test_pred = (model(x_test_tensor)).detach().numpy()
		y_train_pred[y_train_pred >= 0] = 1
		y_train_pred[y_train_pred < 0] = -1
		numWrong = np.sum(np.abs(y_train_pred-Y_train) / 2)
		train_err = numWrong/Y_train.shape[0]
		
		y_test_pred[y_test_pred >= 0] = 1
		y_test_pred[y_test_pred < 0] = -1
		numWrong = np.sum(np.abs(y_test_pred-Y_test) / 2)
		test_err = numWrong/Y_test.shape[0]

		print(f"{d}\t{w}\t{train_err:.8f}\t{test_err:.8f}")
