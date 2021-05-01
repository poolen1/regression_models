
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


user_data_file = "./data/profile.json"
promo_data_file = "./data/portfolio.json"
transaction_data_file = "./data/transcript.json"

user_data = pd.read_json(user_data_file, lines=True)
promo_data = pd.read_json(promo_data_file, lines=True)
transaction_data = pd.read_json(transaction_data_file, lines=True)

print(user_data.to_string())
print(promo_data.to_string())
print(transaction_data.to_string())

# EOF
