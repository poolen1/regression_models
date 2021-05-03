
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def find_user(user_id, data):

    the_user = data.loc[data['id'] == user_id]

    return the_user


def find_promo(promo_id):
    pass


user_data_file = "./data/profile.json"
promo_data_file = "./data/portfolio.json"
transaction_data_file = "./data/transcript.json"

user_data = pd.read_json(user_data_file, lines=True)
promo_data = pd.read_json(promo_data_file, lines=True)
transaction_data = pd.read_json(transaction_data_file, lines=True)

transactions = pd.DataFrame();
offer_completed = pd.DataFrame();
offer_received = pd.DataFrame();

data_set = pd.DataFrame(columns=['amount', 'gender', 'user_age', 'account_age', 'user_income', 'transaction_time',
                                 'offers', 'offer_type', 'offer_diff', 'offer_reward', 'percent_complete', 'channels'])


# Get all 'transaction' type rows
transactions = transaction_data.loc[transaction_data['event'] == 'transaction']

# Get all completed offer evens
offer_completed = transaction_data.loc[transaction_data['event'] == 'offer completed']
# Sort 
transactions = transactions.append(offer_completed)

transactions = transactions.sort_index()



offer_received = transaction_data.loc[transaction_data['event'] == 'offer received']

# transactions.to_csv(r'./data/transactions.csv')
# offer_received.to_csv(r'./date/offer_received.csv')
# print(offer_completed, len(offer_completed))
# print(offer_received.to_string())
# print(user_data.to_string())
# print(promo_data.to_string())
# print(transaction_data.to_string())

# print(transactions.to_string())

# EOF
