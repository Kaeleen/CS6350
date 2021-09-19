import pandas as pd 
import ID3 as id3 
import numpy as np

# set column names and data types

column_names = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'y']
types = {'age': int, 
		'job': str, 
		'marital': str, 
		'education': str,
		'default': str,
		'balance': int,
		'housing': str,
		'loan': str,
		'contact': str,
		'day': int,
		'month': str,
		'duration': int,
		'campaign': int,
		'pdays': int,
		'previous': int,
		'poutcome': str,
		'y': str}

# load train data 
train_data =  pd.read_csv('./bank/train.csv', names=column_names, dtype=types)
train_size = len(train_data)

# load test data 
test_data =  pd.read_csv('./bank/test.csv', names=column_names, dtype=types)
test_size = len(test_data)

# convert numeric features to binary 

numeric_features = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']

for f in numeric_features:
	train_median = train_data[f].median()
	train_data[f] = train_data[f].gt(train_median).astype(int)

	test_median = test_data[f].median()
	test_data[f] = test_data[f].gt(test_median).astype(int)

# set features_dict and label_dict
# treat 'unknown' as an attribute value 
# features_dict = {'age': [0, 1], 
# 		'job': ['admin.', 'unknown', 'unemployed', 'management', 'housemaid', 'entrepreneur', 'student', 'blue-collar', 'self-employed', 'retired', 'technician', 'services'], 
# 		'marital': ['married','divorced','single'], 
# 		'education': ['unknown', 'secondary', 'primary', 'tertiary'],
# 		'default': ['yes', 'no'],
# 		'balance': [0, 1],  
# 		'housing': ['yes', 'no'],
# 		'loan': ['yes', 'no'],
# 		'contact': ['unknown', 'telephone', 'cellular'],
# 		'day': [0, 1],  
# 		'month': ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'],
# 		'duration': [0, 1],  
# 		'campaign': [0, 1],  
# 		'pdays': [0, 1],  
# 		'previous': [0, 1], 
# 		'poutcome': ['unknown', 'other', 'failure', 'success']}

# replace 'unknown' with other values

train_data = train_data.replace(['unknown'], np.nan)
train_data = train_data.fillna(train_data.mode().iloc[0])

test_data = test_data.replace(['unknown'], np.nan)
test_data = test_data.fillna(test_data.mode().iloc[0])

features_dict = {'age': [0, 1], 
		'job': ['admin.', 'unemployed', 'management', 'housemaid', 'entrepreneur', 'student', 'blue-collar', 'self-employed', 'retired', 'technician', 'services'], 
		'marital': ['married','divorced','single'], 
		'education': [ 'secondary', 'primary', 'tertiary'],
		'default': ['yes', 'no'],
		'balance': [0, 1],  
		'housing': ['yes', 'no'],
		'loan': ['yes', 'no'],
		'contact': ['telephone', 'cellular'],
		'day': [0, 1],  
		'month': ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'],
		'duration': [0, 1],  
		'campaign': [0, 1],  
		'pdays': [0, 1],  
		'previous': [0, 1], 
		'poutcome': ['other', 'failure', 'success']}


label_dict = {'y': ['yes', 'no']}

train_res = [[0 for _ in range(3)] for _ in range(16)]
test_res = [[0 for _ in range(3)] for _ in range(16)]


for option in range(3):
	for max_depth in range(16):
		
		dt_generator = id3.ID3(option, max_depth+1)
		
		dt_construction = dt_generator.construct_dt(train_data, features_dict, label_dict)
		
		# prediction results for train data and test data
		train_data['pred_y']= dt_generator.predict(dt_construction, train_data)
		train_data['result'] = (train_data[['y']].values == train_data[['pred_y']].values).all(axis=1).astype(int)
		prediction_train = len(train_data[train_data['result'] == 1]) / train_size
		train_res[max_depth][option] = prediction_train
		
		test_data['pred_y']= dt_generator.predict(dt_construction, test_data)
		test_data['result'] = (test_data[['y']].values == test_data[['pred_y']].values).all(axis=1).astype(int)
		prediction_test = len(test_data[test_data['result'] == 1]) / test_size
		test_res[max_depth][option] = prediction_test


print("results for train data: ")
print(train_res)
print()
print("results for test data: ")
print(test_res)

