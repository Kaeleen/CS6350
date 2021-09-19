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

train_res = [[0 for x in range(6)] for y in range(3)]
test_res = [[0 for x in range(6)] for y in range(3)]


for option in range(3):
	for max_depth in range(6):
		
		dt_generator = id3.ID3(option=option, max_depth=max_depth+1)
		
		dt_construction = dt_generator.construct_dt(train_data, features_dict, label_dict)
		
		# prediction results for train data and test data
		train_data['label2']= dt_generator.predict(dt_construction, train_data)
		train_res[option][max_depth] = train_data.apply(lambda row: 1 if row['label'] == row['label2'] else 0, axis=1).sum() / train_size
		
		test_data['label2']= dt_generator.predict(dt_construction, test_data)
		test_res[option][max_depth] = test_data.apply(lambda row: 1 if row['label'] == row['label2'] else 0, axis=1).sum() / test_size

print("results for train data: ")
print(train_res)
print()
print("results for test data: ")
print(test_res)
