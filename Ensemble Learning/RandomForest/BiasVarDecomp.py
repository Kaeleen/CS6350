import pandas as pd 
import ID3 as id3 
import numpy as np
import matplotlib.pyplot as plt 
import sys



def category_to_numerical_features(df, numerical_features):
	for f in numerical_features:
		median_val = df[f].median()
		df[f] = df[f].gt(median_val).astype(int)

	return df 


def get_bank_data():

	column_names = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'y']
	types = {'age': int, 'job': str, 'marital': str, 'education': str,'default': str,'balance': int,'housing': str,'loan': str,'contact': str,'day': int,'month': str,'duration': int, \
			'campaign': int,'pdays': int,'previous': int,'poutcome': str,'y': str}

	# load train data 
	train_data =  pd.read_csv('./bank/train.csv', names=column_names, dtype=types)
	# load test data 
	test_data =  pd.read_csv('./bank/test.csv', names=column_names, dtype=types)

	numerical_features = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']

	train_data = category_to_numerical_features(train_data, numerical_features)
	test_data = category_to_numerical_features(test_data, numerical_features)

	features_dict = {}
	features_dict['age'] = [0, 1]
	features_dict['job'] = ['admin.', 'unknown', 'unemployed', 'management', 'housemaid', 'entrepreneur', 'student', 'blue-collar', 'self-employed', 'retired', 'technician', 'services']
	features_dict['marital'] = ['married','divorced','single']
	features_dict['education'] = ['unknown', 'secondary', 'primary', 'tertiary']
	features_dict['default'] = ['yes', 'no']
	features_dict['balance'] = [0, 1]
	features_dict['housing'] = ['yes', 'no']
	features_dict['loan'] = ['yes', 'no']
	features_dict['contact'] = ['unknown', 'telephone', 'cellular']
	features_dict['day'] = [0, 1]
	features_dict['month'] = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
	features_dict['duration'] = [0, 1]
	features_dict['campaign'] = [0, 1]
	features_dict['pdays'] = [0, 1]
	features_dict['previous'] = [0, 1]
	features_dict['poutcome'] = ['unknown', 'other', 'failure', 'success']
	
	label_dict = {}
	label_dict['y'] = ['yes', 'no']

	return features_dict, label_dict, train_data, test_data

def process_data(features_dict, label_dict, train_data, test_data, num_subset, label_name):
	
	train_size, test_size = len(train_data),len(test_data)
	test_py = np.array([[0 for x in range(test_size)] for y in range(100)])
	test_py_first = np.array([0 for x in range(test_size)])

	for i in range(100):
		train_subset = train_data.sample(n=1000, replace=False, random_state=i)
		for t in range(500):
			sampled = train_subset.sample(frac=0.01, replace=True, random_state=t)

			# build tree 
			dt_generator = id3.ID3(option=0, max_depth=15, subset = num_subset)
			
			dt_construction = dt_generator.construct_dt(sampled, features_dict, label_dict)

			# predict test 
			pred_label = dt_generator.predict(dt_construction, test_data)
			pred_label = np.array(pred_label.tolist())

			pred_label[pred_label == 'yes'] = 1 
			pred_label[pred_label == 'no'] = -1
			pred_label = pred_label.astype(int)
			test_py[i] = test_py[i]+pred_label
			if t==0:
				test_py_first = test_py_first+pred_label


	true_y = np.array(test_data[label_name].tolist())
	true_y[true_y=='yes'] = 1
	true_y[true_y=='no'] = -1
	true_y=true_y.astype(int)

	# predicts first tree 
	test_py_first = test_py_first/100 

	# bias 
	bias = np.mean(np.square(test_py_first - true_y))

	#variance 
	mean = np.mean(test_py_first)
	variance = np.sum(np.square(test_py_first - mean)) / (test_size - 1)
	squaredError = bias+variance

	print("Decomposition results when subset =", num_subset)
	print("100 single tree predictor: ", bias, variance, squaredError)

	# random forest 
	test_py = np.sum(test_py,axis=0) / 50000

	# bias 
	bias = np.mean(np.square(test_py - true_y))

	# variance 
	mean = np.mean(test_py)
	variance = np.sum(np.square(test_py - mean)) / (test_size - 1)
	squaredError = bias+variance 
	print("100 random forest predictor: ", bias, variance, squaredError)

num_subset = int(sys.argv[1])
features_dict, label_dict, train_data, test_data = get_bank_data()
process_data(features_dict, label_dict, train_data, test_data, num_subset, 'y')