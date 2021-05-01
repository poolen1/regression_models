
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

"""
for i in range(len(transaction_data)):
    # print(transaction_data.loc[i]['event'])
    if transaction_data.loc[i]['event'] == "transaction":
        print(transaction_data.loc[i])
"""

for i in range(len(transaction_data)):
    if transaction_data.loc[i]["event"] == "offer completed":
        offer_completed.append(transaction_data.loc[i-1])
        offer_completed.append(transaction_data.loc[i])
    elif transaction_data.loc[i]["event"] == "offer received":
        offer_received.append(transaction_data.loc[i])

print(offer_completed.to_string())
# print(user_data.to_string())
# print(promo_data.to_string())
# print(transaction_data.to_string())

# EOF
