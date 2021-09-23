import pandas as pd 
import ID3 as id3 
import numpy as np
import sys


def get_car_data():
	
	# load data
	column_names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'label']
	types = {'buying': str, 'maint': str, 'doors': str, 'persons': str, 'lug_boot': str, 'safety': str, 'label': str}
	# load train data
	train_data =  pd.read_csv('./car/train.csv', names=column_names, dtype=types)
	
	# load test data
	test_data =  pd.read_csv('./car/test.csv', names=column_names, dtype=types)


	# get features_dict and label_dict
	features_dict = {}
	label_dict = {}
	features_dict['buying'] = ['vhigh', 'high', 'med', 'low']
	features_dict['maint'] = ['vhigh', 'high', 'med', 'low']
	features_dict['doors'] = ['2', '3', '4', '5more']
	features_dict['persons'] = ['2', '4', 'more']
	features_dict['lug_boot'] = ['small', 'med', 'big']
	features_dict['safety'] = ['low', 'med', 'high']


	label_dict['label']= ['unacc', 'acc', 'good', 'vgood']

	return features_dict, label_dict, train_data, test_data

def category_to_numerical_features(df, numerical_features):
	for f in numerical_features:
		median_val = df[f].median()
		df[f] = df[f].gt(median_val).astype(int)

	return df 

def replace_unknown_values(df):
	df = df.replace(['unknown'], np.nan)
	df = df.fillna(df.mode().iloc[0])
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

def process_res(features_dict, label_dict, train_data, test_data, depth_range, label_name):
	
	train_size = len(train_data.index)
	test_size = len(test_data.index)

	train_res = [[0 for _ in range(3)] for _ in range(depth_range)]
	test_res = [[0 for _ in range(3)] for _ in range(depth_range)]


	for option in range(3):
		for max_depth in range(depth_range):
			
			dt_generator = id3.ID3(option, max_depth+1)
			
			dt_construction = dt_generator.construct_dt(train_data, features_dict, label_dict)
			
			# prediction results for train data and test data
			train_data['pred_label']= dt_generator.predict(dt_construction, train_data)
			train_data['result'] = (train_data[[label_name]].values == train_data[['pred_label']].values).all(axis=1).astype(int)
			prediction_train = len(train_data[train_data['result'] == 1]) / train_size
			train_res[max_depth][option] = round(prediction_train,3)
			
			test_data['pred_label']= dt_generator.predict(dt_construction, test_data)
			test_data['result'] = (test_data[[label_name]].values == test_data[['pred_label']].values).all(axis=1).astype(int)
			prediction_test = len(test_data[test_data['result'] == 1]) / test_size
			test_res[max_depth][option] = prediction_test

	print("results for train data: ")
	print(train_res)
	print()
	print("results for test data: ")
	print(test_res)


questionDescription = sys.argv[1]

if questionDescription == 'car':
	features_dict, label_dict, train_data, test_data = get_car_data()
	process_res(features_dict, label_dict, train_data, test_data, 6, 'label')

if questionDescription == 'bank':
	features_dict, label_dict, train_data, test_data = get_bank_data()
	process_res(features_dict, label_dict, train_data, test_data, 16, 'y')

if questionDescription == 'bank_replace_unknowns':
	features_dict, label_dict, train_data, test_data = get_bank_data()
	train_data = replace_unknown_values(train_data)
	test_data = replace_unknown_values(test_data)
	process_res(features_dict, label_dict, train_data, test_data, 16, 'y')


