import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from jupyterthemes import jtplot
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error


jtplot.style(theme="monokai")

data = pd.read_csv("data/reg_demo_data.csv")
data.head()
data.drop(list(data.columns)[0], axis=1, inplace=True)
data.head()
print(list(data.columns))


X = np.array(data['x']).reshape(-1,1)
Y = np.array(data['y']).reshape(-1,1)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size= 0.2, random_state=5)

# Linear Regression Model
model = LinearRegression()
model.fit(X_train,Y_train)
y_train_pred = model.predict(X_train)



# Regression Assessment
rmse = np.sqrt(mean_squared_error(Y_train, y_train_pred))
r2 = round(model.score(X_train, Y_train), 2)

from IPython.display import display, Markdown
display(Markdown("The Root $\\text{Mean}^2 $ Error is" + " {}".format(rmse)))
display(Markdown(f"$R^2$ = {r2}"))

# Optimized Parameters
display(Markdown("$ \\beta_1 = $" + "{}".format(model.coef_[0])))
display(Markdown("$ \\beta_0 = $" + "{}".format(model.intercept_)))

# Visualizations
plt.figure(figsize=(10,5))
plt.scatter(X_train, Y_train)
plt.plot(X_train, model.predict(X_train))
plt.title("Linear Regression Plot of Celsius to Farenheit", size=30)
plt.xlabel(r"$\degree C$", size=24)
plt.ylabel(r"$\degree F$", size=24)
plt.savefig("images/sk_regression_plot.png")
