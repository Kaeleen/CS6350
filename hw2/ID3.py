import pandas as pd
import math
import copy

# TreeNode class 

class TreeNode:
	def __init__(self):
		self.feature = None
		self.children = None
		self.depth = -1
		self.is_leaf_node = False
		self.label = None
	
	# functions to set values
	def set_feature(self, feature):
		self.feature = feature

	def set_children(self, children):
		self.children = children

	def set_depth(self, depth):
		self.depth = depth

	def set_leaf(self, status):
		self.is_leaf_node = status

	def set_label(self, label):
		self.label = label

	# functions to return values
	def is_leaf(self):
		return self.is_leaf_node

	def get_depth(self):
		return self.depth

	def get_label(self):
		return self.label 


# algorithum implementation

class ID3:
	# Option 0: Entropy, option 1: ME, Option 2: GI
	def __init__(self, option=1, max_depth = 10):
		self.option = option
		self.max_depth = max_depth
	
	
	def set_max_depth(self, max_depth):
		self.max_depth = max_depth


	def set_option(self, option):
		self.option = option

	def calc_entropy(self, data, label_dict):
		"""
		This function returns entropy for a specific data subsets and a set of labels
		"""
		label_key = list(label_dict.keys())[0]
		label_values = label_dict[label_key]
		if len(data) == 0:
			return 0
		entropy = 0
		for value in label_values:
			p = len(data[data[label_key] == value]) / len(data)
			if p != 0:
				entropy += -p * math.log2(p)
		return entropy
	
	def calc_ME(self, data, label_dict):
		"""
		This function returns ME for a specific data subsets and a set of labels
		"""
		label_key = list(label_dict.keys())[0]
		label_values = label_dict[label_key]
		if len(data) == 0:
			return 0
		max_p = 0
		for value in label_values:
			p = len(data[data[label_key] == value]) / len(data)
			max_p = max(max_p, p)
		return 1 - max_p
		
	
	def calc_GI(self, data, label_dict):
		"""
		This function returns GI for a specific data subsets and a set of labels
		"""
		label_key = list(label_dict.keys())[0]
		label_values = label_dict[label_key]
		if len(data) == 0:
			return 0
		temp = 0
		for value in label_values:
			p = len(data[data[label_key] == value]) / len(data)
			temp += p**2
		return 1 - temp
	

	def get_majority_label(self,column):
		"""
		This function returns the major label
		"""

		majority_label = column.value_counts().idxmax()

		return majority_label

	def get_heuristics(self):

		if self.option == 0:
			heuristics = self.calc_entropy
		if self.option == 1:
			heuristics = self.calc_ME
		if self.option == 2:
			heuristics = self.calc_GI

		return heuristics


	def get_feature_with_max_gain(self, data, label_dict, features_dict):

		heuristics = self.get_heuristics()
		measure = heuristics(data, label_dict)

		max_gain = float('-inf')
		max_f_name = ''

		for f_name, f_values in features_dict.items():
			gain = 0
			for val in f_values:
				subset = data[data[f_name] == val]
				p = len(subset.index) / len(data)
				
				gain += p * heuristics(subset, label_dict)

			# get maximum gain and feature name	
			gain = measure - gain
			if gain > max_gain:
				max_gain = gain
				max_f_name = f_name

		return max_f_name
		

	def best_feature_split(self, cur_node):
		next_nodes = []
		features_dict = cur_node['features_dict']
		label_dict = cur_node['label_dict']
		dt_node = cur_node['dt_node']
		data = cur_node['data']

		
		label_key = list(label_dict.keys())[0]
		label_values = label_dict[label_key]
		
		if len(data) > 0:
			majority_label = self.get_majority_label(data[label_key])
			
		heuristics = self.get_heuristics()
		measure = heuristics(data, label_dict)

		# check leaf nodes
		if measure == 0 or dt_node.get_depth() == self.max_depth or len(features_dict) == 0:
			dt_node.set_leaf(True)
			if len(data) > 0:
				dt_node.set_label(majority_label)
			return next_nodes

		
		children = {}
		max_f_name = self.get_feature_with_max_gain(data, label_dict, features_dict)
		dt_node.set_feature(max_f_name)

		# remove the feature that has been splitted on, get remaining features
		rf = copy.deepcopy(features_dict)
		rf.pop(max_f_name, None)
	
		for val in features_dict[max_f_name]:
			child_node = TreeNode()
			child_node.set_label(majority_label)
			child_node.set_depth(dt_node.get_depth() + 1)
			children[val] = child_node
			primary_node = {'data': data[data[max_f_name] == val],'features_dict': rf, 'label_dict': label_dict, 'dt_node': child_node}
			next_nodes.append(primary_node)
		
		# set chiildren nodes
		dt_node.set_children(children)
		
		return next_nodes
	   
	
	# construct the decision tree
	def construct_dt(self, data, features_dict, label_dict):

		# bfs using queue
		import queue
		dt_root = TreeNode()
		dt_root.set_depth(0)
		root = {'data': data,'features_dict': features_dict, 'label_dict': label_dict, 'dt_node': dt_root}

		Q = queue.Queue()
		Q.put(root)
		while not Q.empty():
			cur_node = Q.get()
			for node in self.best_feature_split(cur_node):
				Q.put(node)
		return dt_root
	

	def classify_one(self, dt, data):
		temp = dt
		while not temp.is_leaf(): 
			temp = temp.children[data[temp.feature]]
		return temp.label

	def predict(self, dt, data):
		return data.apply(lambda row: self.classify_one(dt, row), axis=1)