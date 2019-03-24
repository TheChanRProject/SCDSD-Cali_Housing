import numpy as np
import pandas as pd

x = np.arange(-100,101)
print(x)

def farenheit(x):
    return 1.8*x + 32

y = farenheit(x)
print(y)

data_dict = {'x': x, 'y':y}
print(data_dict)

df = pd.DataFrame.from_dict(data_dict)
df.head()
df.to_csv("data/reg_demo_data.csv")
