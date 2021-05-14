
import pandas as pd
import time
import numpy as np


start_time = time.time()

print("Start Time: ", start_time)

user_data_file = "./data/profile.json"
promo_data_file = "./data/portfolio.json"
transaction_data_file = "./data/transcript.json"

user_data = pd.read_json(user_data_file, lines=True)
promo_data = pd.read_json(promo_data_file, lines=True)
transaction_data = pd.read_json(transaction_data_file, lines=True)

transactions = pd.DataFrame()
offer_completed = pd.DataFrame()
offer_received = pd.DataFrame()

# data_set = pd.DataFrame(columns=['amount', 'gender', 'user_age', 'user_income', 'transaction_time',
#                                  'offer_type', 'offer_diff', 'offer_reward', 'percent_complete', 'channels'])
data_set = pd.DataFrame()

# Get all 'transaction' type rows
transactions = transaction_data.loc[transaction_data['event'] == 'transaction']

# Get all completed offer events
offer_completed = transaction_data.loc[transaction_data['event'] == 'offer completed']
transactions = transactions.append(offer_completed)
transactions = transactions.sort_index()

data_set = transactions

# User Data
data_set['gender'] = data_set['person'].map(user_data.set_index('id')['gender'])
data_set['user_age'] = data_set['person'].map(user_data.set_index('id')['age'])
data_set['user_income'] = data_set['person'].map(user_data.set_index('id')['income'])

# Normalize user data
max_age = data_set['user_age'].max()
data_set['user_age'] = data_set['user_age']/max_age
data_set['user_age'] = data_set['user_age'].round(3)

max_income = data_set['user_income'].max()
data_set['user_income'] = data_set['user_income']/max_income
data_set['user_income'] = data_set['user_income'].round(3)

# Promo Data
data_set['offer_id'] = [d.get('offer_id') for d in transactions.value]
data_set['offer_id'] = data_set['offer_id'].shift(periods=-1)
data_set['amount'] = [d.get('amount') for d in transactions.value]
data_set = data_set[data_set.event != 'offer completed']

data_set['offer_type'] = data_set['offer_id'].map(promo_data.set_index('id')['offer_type']).astype('category')
data_set['offer_diff'] = data_set['offer_id'].map(promo_data.set_index('id')['difficulty'])
data_set['offer_reward'] = data_set['offer_id'].map(promo_data.set_index('id')['reward'])

offer_channels = data_set['offer_id'].map(promo_data.set_index('id')['channels']).apply(pd.Series)

data_set['web'] = offer_channels[0].apply(lambda i: i if i == 0 else 1)
data_set['email'] = offer_channels[1].apply(lambda i: i if i == 0 else 1)
data_set['mobile'] = offer_channels[2].apply(lambda i: i if i == 0 else 1)
data_set['social'] = offer_channels[3].apply(lambda i: i if i == 0 else 1)

# Normalize promo data
max_diff = data_set['offer_diff'].max()
data_set['offer_diff'] = data_set['offer_diff']/max_diff
data_set['offer_diff'] = data_set['offer_diff'].round(3)

max_reward = data_set['offer_reward'].max()
data_set['offer_reward'] = data_set['offer_reward']/max_reward
data_set['offer_reward'] = data_set['offer_reward'].round(3)

del data_set['person']
del data_set['offer_id']
del data_set['value']
del data_set['event']

data_set = data_set[['amount', 'time', 'gender', 'user_age', 'user_income', 'offer_type', 'offer_diff',
                     'offer_reward', 'web', 'email', 'mobile', 'social']]

data_set['gender'] = data_set['gender'].replace([np.nan, 'M', 'F', 'O'], [0, 1, 2, 3])
data_set['offer_type'] = data_set['offer_type'].replace([np.nan, 'bogo', 'discount', 'informational'], [0, 1, 2, 3])
data_set = data_set.fillna(0)

# data_set.reset_index(drop=True, inplace=True)

data_set.to_csv('./data/data_set.csv')

"""
y_list = data_set.loc[:, 'amount']
X_list = data_set.drop('amount', axis=1)

# X and y, training and test data
y = y_list.values
X = np.array(X_list)
"""

# print(data_set.to_string())
# print(y)
# print(X)
# print("Length: ", len(data_set))

end_time = time.time() - start_time

print("Runtime: ", end_time)

# EOF
