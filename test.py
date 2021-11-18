import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('./data/bouy_data_daily.csv')

print(df.head())
sns_plot = sns.heatmap(df.isnull(), cbar=False)

sns_plot.figure.savefig("output.png")


# make a metric of missing percentage for wave height and wave period, train main data using least missing

