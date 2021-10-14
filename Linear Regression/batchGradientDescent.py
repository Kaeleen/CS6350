import numpy as np 
import math
from numpy import linalg as LA


def cost_function(X, Y, W):
	res = 0 
	for i in range(len(X)):
		temp = (Y[i] - np.dot(W, X[i]))**2 
		res += temp 
	return 0.5*res


def LMS_gradiant(X, Y, r):
	costs = []  

	W = np.zeros(X.shape[1])


	e = math.inf

	while e > 10e-6:
		grad_w = np.zeros(X.shape[1])
		
		for j in range(len(X[0])):
			temp = 0 
			for i in range(len(X)):
				temp += X[i][j] *(Y[i] - np.dot(W, X[i]))
			grad_w[j] = temp 

		new_W = W + r*grad_w

		e = LA.norm(W - new_W)
		costs.append(cost_function(X, Y, W))

		W = new_W

	costs.append(cost_function(X, Y, W))
	return W, costs



