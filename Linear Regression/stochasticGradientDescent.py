import numpy as np 
import math
from numpy import linalg as LA
import random


def cost_function(X, Y, W):
	res = 0 
	for i in range(len(X)):
		temp = (Y[i] - np.dot(W, X[i]))**2 
		res += temp 
	return 0.5*res

def sgd(X, Y, r):

	W = np.zeros(X.shape[1])


	e = math.inf

	costs = [cost_function(X, Y, W)]

	while e > 10e-10:
		i = random.randrange(len(X))

		grad_w = np.zeros(X.shape[1])
		for j in range(len(X[0])): 
			grad_w[j] = X[i][j] *(Y[i] - np.dot(W, X[i]))

		new_W = W + r*grad_w
		W = new_W
		new_cost = cost_function(X, Y, W) 
		e = abs(new_cost - costs[-1])
		costs.append(new_cost)

	return W, costs

