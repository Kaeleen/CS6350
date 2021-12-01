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

def sgd(x, y, nn, GammaSchedule, T):
	# initialize weights
	idxs = np.arange(x.shape[0])
	gamma = GammaSchedule.gamma0
	iterations = 1
	
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

	return deepcopy(nn)


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