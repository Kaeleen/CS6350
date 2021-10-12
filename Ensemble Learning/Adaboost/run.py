import pandas as pd 
import numpy as np 
import modifiedID3 as dt 
import math 
import matplotlib.pyplot as plt 


def category_to_numerical_features(df, numerical_features):
	for f in numerical_features:
		median_val = df[f].median()
		df[f] = df[f].gt(median_val).astype(int)

	return df 


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

T = 500

train_size, test_size = len(train_data),len(test_data)
alphas = [0 for x in range(T)]
weights = np.array([1/train_size for x in range(train_size)])
# print(weights)

train_errors, test_errors = [0 for x in range(T)], [0 for x in range(T)]
train_errorsT, test_errorsT = [0 for x in range(T)], [0 for x in range(T)]

test_py = np.array([0 for x in range(test_size)])

train_py = np.array([0 for x in range(train_size)])
for t in range(T):
	dt_generator = dt.MID3(option=0, max_depth=1)
			
	dt_construction = dt_generator.construct_dt(train_data, features_dict, label_dict, weights)

	# train errors
	train_data['pred_label']= dt_generator.predict(dt_construction, train_data)
	train_data['result'] = (train_data[['y']].values == train_data[['pred_label']].values).all(axis=1).astype(int)
	err = 1 - len(train_data[train_data['result'] == 1]) / train_size
	train_errors[t] = err

	# test errors
	test_data['pred_label']= dt_generator.predict(dt_construction, test_data)
	test_data['result'] = (test_data[['y']].values == test_data[['pred_label']].values).all(axis=1).astype(int)
	test_errors[t] = 1 - len(test_data[test_data['result'] == 1]) / test_size
	
	# weighted errors and alphas
	temp = train_data.apply(lambda row: 1 if row['y'] == row['pred_label'] else -1, axis=1)
	temp = np.array(temp.tolist())
	w = weights[temp == -1]
	err = np.sum(w)

	alpha = 0.5 * math.log((1-err)/err)
	alphas[t] = alpha 

	# get new weights 
	weights = np.exp(temp * -alpha) * weights
	total = np.sum(weights)
	weights = weights/total

	#errors of all decision stumps

	pred_label = np.array(train_data['pred_label'].tolist())
	pred_label[pred_label == 'yes'] = 1 
	pred_label[pred_label == 'no'] = -1
	pred_label = pred_label.astype(int)
	train_py = train_py+pred_label*alpha
	pred_label = pred_label.astype(str)
	pred_label[train_py > 0] = 'yes'
	pred_label[train_py <= 0] = 'no'
	train_data['pred_label'] = pd.Series(pred_label)

	train_data['result'] = (train_data[['y']].values == train_data[['pred_label']].values).all(axis=1).astype(int)
	
	train_errorsT[t] = 1 - len(train_data[train_data['result'] == 1]) / train_size


	#  test data 
	
	pred_label = np.array(test_data['pred_label'].tolist())
	pred_label[pred_label == 'yes'] = 1 
	pred_label[pred_label == 'no'] = -1
	pred_label = pred_label.astype(int)
	test_py = test_py+pred_label*alpha
	pred_label = pred_label.astype(str)
	pred_label[test_py > 0] = 'yes'
	pred_label[test_py <= 0] = 'no'
	test_data['pred_label'] = pd.Series(pred_label)

	test_data['result'] = (test_data[['y']].values == test_data[['pred_label']].values).all(axis=1).astype(int)
	
	test_errorsT[t] = 1 - len(test_data[test_data['result'] == 1]) / test_size



fig, (ax1, ax2) = plt.subplots(2, 1)
fig.tight_layout(h_pad=2)

ax1.plot(train_errors,  color='blue', label='train_data')
ax1.set_ylabel('error rate')
ax1.plot(test_errors,  color='red', label='test_data')

ax1.set_title("Individual tree prediction results")
ax1.legend()

ax2.plot(train_errorsT,  color='blue', label='train_data')
ax2.set_ylabel('error rate')
ax2.set_xlabel('T')
ax2.plot(test_errorsT,  color='red', label='test_data')

ax2.set_title("All decision trees prediction results")
ax2.legend()

fig.savefig('Adaboost.png', dpi=300, bbox_inches='tight')




