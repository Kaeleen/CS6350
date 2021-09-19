import ID3 as id3
import pandas as pd


# load data
column_names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'label']
types = {'buying': str, 'maint': str, 'doors': str, 'persons': str, 'lug_boot': str, 'safety': str, 'label': str}
# load train data
train_data =  pd.read_csv('./car/train.csv', names=column_names, dtype=types)
train_size = len(train_data.index)
# load test data
test_data =  pd.read_csv('./car/test.csv', names=column_names, dtype=types)
test_size = len(test_data.index)

# get features_dict and label_dict
features_dict = {'buying': ['vhigh', 'high', 'med', 'low'], 
			'maint':  ['vhigh', 'high', 'med', 'low'], 
			'doors':  ['2', '3', '4', '5more'], 
			'persons': ['2', '4', 'more'], 
			'lug_boot': ['small', 'med', 'big'],  
			'safety':  ['low', 'med', 'high']  }

label_dict = {'label': ['unacc', 'acc', 'good', 'vgood']}

train_res = [[0 for _ in range(3)] for _ in range(6)]
test_res = [[0 for _ in range(3)] for _ in range(6)]


for option in range(3):
	for max_depth in range(6):
		
		dt_generator = id3.ID3(option, max_depth+1)
		
		dt_construction = dt_generator.construct_dt(train_data, features_dict, label_dict)
		
		# prediction results for train data and test data
		train_data['pred_label']= dt_generator.predict(dt_construction, train_data)
		train_data['result'] = (train_data[['label']].values == train_data[['pred_label']].values).all(axis=1).astype(int)
		prediction_train = len(train_data[train_data['result'] == 1]) / train_size
		train_res[max_depth][option] = round(prediction_train,3)
		
		test_data['pred_label']= dt_generator.predict(dt_construction, test_data)
		test_data['result'] = (test_data[['label']].values == test_data[['pred_label']].values).all(axis=1).astype(int)
		prediction_test = len(test_data[test_data['result'] == 1]) / test_size
		test_res[max_depth][option] = prediction_test

print("results for train data: ")
print(train_res)
print()
print("results for test data: ")
print(test_res)
