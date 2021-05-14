
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('data_set.csv')

corr = data.corr()

ax = sns.heatmap(
    corr,
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
)

plt.show()