
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


user_data_file = "./data/profile.json"
promo_data_file = "./data/portfolio.json"
transaction_data_file = "./data/transcript.json"

user_data = pd.read_json(user_data_file, lines=True)
promo_data = pd.read_json(promo_data_file, lines=True)
transaction_data = pd.read_json(transaction_data_file, lines=True)

offer_completed = pd.DataFrame();
offer_received = pd.DataFrame();

offer_completed = transaction_data.loc[transaction_data['event'].shift(-1) == 'offer completed']
offer_completed = offer_completed.loc[offer_completed['event'] == 'transaction']
offer_completed = offer_completed.append(transaction_data.loc[transaction_data['event'] == 'offer completed'])

offer_completed = offer_completed.sort_index()

offer_received = transaction_data.loc[transaction_data['event'] == 'offer received']

offer_completed.to_csv(r'./data/offer_completed.csv')
# offer_received.to_csv(r'./date/offer_received.csv')
# print(offer_completed, len(offer_completed))
# print(offer_received.to_string())
# print(user_data.to_string())
# print(promo_data.to_string())
# print(transaction_data.to_string())

# EOF
