import numpy as np 
import csv
from math import exp
from dataclasses import dataclass
from copy import deepcopy

@dataclass
class GammaSchedule:
	gamma0: float
	d: float


class NeuralNetwork:
	def __init__(self, layers, numInputs, hiddenNodeCount, randInit):
		self.layerCount = layers
	   
		self.layerNodeCounts = np.concatenate([np.array([numInputs]), np.array(hiddenNodeCount)+1, np.array([2])])
	  
		self.nodes = np.zeros((layers, np.amax(self.layerNodeCounts)))
		self.nodes[:,0] = np.ones(layers) 

		self.weights = np.zeros((layers, np.amax(self.layerNodeCounts), np.amax(self.layerNodeCounts)))
		if randInit == True:
			self.weights = np.random.normal(size=(layers,np.amax(self.layerNodeCounts),np.amax(self.layerNodeCounts)))
	
		self.dweights = np.zeros((layers,np.amax(self.layerNodeCounts),np.amax(self.layerNodeCounts)))
		self.y = None

def sigmoid(x):
	return 1 / (1 + exp(-1*x))


def sigmoid_deriv(x):
	return x * (1 - x)


def backward(y, nn):
	
	dLdy = nn.y - y
	cache = np.zeros((len(nn.layerNodeCounts), np.amax(nn.layerNodeCounts)))

	for target in reversed(range(1, len(nn.layerNodeCounts))):
		if target != 0 and target == nn.layerCount: 
			for to in range(1, nn.layerNodeCounts[target]):
				cache[target, to] = dLdy
				for fromNode in range(nn.layerNodeCounts[target-1]):
					nn.dweights[target-1,to,fromNode] = cache[target, to] * nn.nodes[target-1, fromNode]
		else: 
			for to in range(1, nn.layerNodeCounts[target]):
				cache[target, to] = 0
				for connected in range(1, nn.layerNodeCounts[target+1]):
					cache[target, to] += cache[target+1, connected] * nn.weights[target, connected, to] * sigmoid_deriv(nn.nodes[target, to])
			# calculate derivatives
			for to in range(nn.layerNodeCounts[target]):
				for fromNode in range(nn.layerNodeCounts[target-1]):
					nn.dweights[target-1,to,fromNode] = cache[target, to] * nn.nodes[target-1, fromNode]


def forward(x, nn):
	nn.nodes[0,:x.shape[1]] = np.copy(x)
	for layer in range(1, len(nn.layerNodeCounts)): 
		for node in range(1, nn.layerNodeCounts[layer]): 
			layerSum = np.sum(np.multiply(nn.nodes[layer-1,:], nn.weights[layer-1,node,:]))
			if layer == nn.layerCount: 
				nn.y = layerSum
			else: 
				nn.nodes[layer, node] = sigmoid(layerSum)

def sgd(x, y, nn, GammaSchedule, T, checkConverge):
	# initialize weights
	idxs = np.arange(x.shape[0])
	gamma = GammaSchedule.gamma0
	iterations = 1
	
	lossList = []
	for epoch in range(T):
		# shuffle data
		np.random.shuffle(idxs)

		for i in idxs:
			# calculate iteration gamma
			gamma = GammaSchedule.gamma0 / (1 + (GammaSchedule.gamma0 * iterations / GammaSchedule.d))

			forward(x[i], nn)
			backward(y[i], nn)
			nn.weights = np.subtract(nn.weights, x.shape[0]*gamma*nn.dweights)

			iterations += 1

		if checkConverge == True:
			lossSum = 0
			for i in idxs:
				forward(x[i], nn)
				lossSum += 0.5 * (nn.y - y[i])**2
			lossList.append(lossSum)

	return deepcopy(nn), lossList


def sgd_predict(x, nn):
	predictions = []
	for ex in x:
		forward(ex, nn)
		p = nn.y
		if p < 0:
			predictions.append(-1)
		else:
			predictions.append(1)
	return np.array(predictions)


# class nn:
# 	def __init__(self, width):
# 		self.in_d = width[0]
# 		self.out_d = width[-1]
# 		self.lr = 0.1
# 		self.d = 0.1
# 		self.epoch = 100
# 		self.gamma = 0.1

# 		# width including input and output
# 		self.width = width
# 		self.layers = len(width)
		
# 		# weights
# 		self.w = [None for _ in range(self.layers)]
# 		self.dw = [None for _ in range(self.layers)]
# 		for i in range(1, self.layers-1):
# 			wi = np.random.normal(0, 1, (self.width[i] - 1, self.width[i-1]))
# 			# wi = np.zeros([self.width[i] - 1, self.width[i-1]])
# 			self.w[i] = wi
# 			self.dw[i] = np.zeros([self.width[i] - 1, self.width[i - 1]])
# 		i = self.layers - 1
# 		wi = np.random.normal(0, 1, (self.width[i], self.width[i-1]))
# 		self.w[i] = wi
# 		self.dw[i] = np.zeros([self.width[i], self.width[i - 1]])
# 		# nodes
# 		self.nodes = [np.ones([self.width[i], 1]) for i in range(self.layers)]
		

# 	def train(self, x, y):
# 		num_sample = x.shape[0]
# 		dim = x.shape[1]
# 		idx = np.arange(num_sample)
# 		for t in range(self.epoch):
# 			np.random.shuffle(idx)
# 			x = x[idx,:]
# 			y = y[idx]
# 			for i in range(num_sample):
# 				self.forward_backward(x[i,:].reshape([self.in_d, 1]), y[i,:].reshape([self.out_d, 1]))
# 				lr = self.gamma / (1 + self.gamma/self.d * t)
# 				self.update_w(lr)

# 	def update_w(self, lr):
# 		for i in range(1, self.layers):
# 			self.w[i] = self.w[i] - self.lr * self.dw[i]
		
# 	def forward(self, x):
# 		# input
# 		self.nodes[0] = x
# 		for i in range(1, self.layers-1):
# 			self.nodes[i][:-1,:] = self.sigmoid(np.matmul(self.w[i], self.nodes[i - 1]).reshape([-1,1]))
# 		# output
# 		i = self.layers - 1
# 		self.nodes[i] = np.matmul(self.w[i], self.nodes[i - 1])
	
# 	def sigmoid(self, x):
# 		from math import exp
# 		return 1 / (1 + exp(-1*x))

# 	def backward(self, y):
# 		# output
# 		# dLdy
# 		dLdz = self.nodes[-1] - y
# 		nk = self.width[-1]
# 		dzdw =  np.transpose(np.tile(self.nodes[-2], [1, nk]))
# 		self.dw[-1] = dLdz * dzdw
# 		# dydz
# 		dzdz = self.w[-1][:, :-1]
# 		# derivative between k-1 and k
# 		for i in reversed(range(1, self.layers - 1)):
# 			# sigmod derivative z = sigmoid(a)
# 			nk = self.width[i] - 1
# 			# ignore bias term in k
# 			z_in = self.nodes[i-1]
# 			z_out = self.nodes[i][:-1]
# 			dadw = np.transpose(np.tile(z_in, [1, nk]))
# 			dzdw = z_out * (1 - z_out) * dadw
# 			dLdz = np.matmul(np.transpose(dzdz), dLdz)
# 			dLdw = dLdz * dzdw
# 			self.dw[i] = dLdw

# 			dzdz = z_out * (1 - z_out) * self.w[i] 
# 			dzdz = dzdz[:, :-1]

# 	def forward_backward(self, x, y):
# 		self.forward(x)
# 		self.backward(y)
	
# 	def fit(self, x):
# 		num_sample = x.shape[0]
# 		l = []
# 		for i in range(num_sample):
# 			self.forward(x[i,:].reshape(self.in_d))
# 			y = self.nodes[-1]
# 			l.append(np.transpose(y))
# 		y_pred = np.concatenate(l, axis=0)
# 		return y_pred