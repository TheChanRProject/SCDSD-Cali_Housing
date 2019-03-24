import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from jupyterthemes import jtplot
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error

data = pd.read_csv("data/reg_demo_data.csv")
data.head()
print(list(data.columns))
data.drop(list(data.columns)[0], axis=1, inplace=True)
data.head()


X = data['x']
Y = data['y']
