import itertools
import yfinance as yf
import investpy            
import yahoo_fin.stock_info as si
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np

data = pd.DataFrame(investpy.get_etf_historical_data(etf='iShares 20+ Year Treasury Bond', country='united states', from_date='01/01/2013', to_date='01/01/2022'))
df = data['Close']

import pandas_datareader as web 
import datetime

ieo = web.get_data_yahoo(['IEO'], start=datetime.datetime(2013, 1, 1), end=datetime.datetime(2022, 1, 1))['Close']

print(ieo.head())

df.to_csv("df.csv")
df = pd.read_csv("df.csv")
print(df.head())

df.index = pd.to_datetime(df['Date'], format='%Y-%m-%d')
del df['Date']

#Data is already well formatted
df.index

#graph data wiht monthly mean
df.sort_index(inplace=True)
import warnings
fig, ax = plt.subplots(figsize=(20, 6))
ax.plot(df,marker='.', linestyle='-', linewidth=0.5, label='Daily')
ax.plot(df.resample('1m').mean(),marker='o', markersize=8, linestyle='-', label='Monthly Mean Resample')
ax.set_ylabel('iShares 20+ Prices')
ax.legend()
plt.show()

#test non-stationarity
from statsmodels.tsa.stattools import adfuller
from numpy import log
result = adfuller(df.dropna())
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])

#Auto Correlation functions
sm.graphics.tsa.plot_acf(df.values.squeeze(), lags=100)
sm.graphics.tsa.plot_pacf(df.values.squeeze(), lags=100)
plt.show()

#decomposition after selecting frequency and filling missing values
df.sort_index(inplace=True)
df = df.asfreq('d')
df = df.fillna(method='bfill').fillna(method='ffill')
df.sort_index(inplace=True)
rcParams['figure.figsize'] = 16, 6
decomposition = sm.tsa.seasonal_decompose(df, model = 'additive')
fig = decomposition.plot()
plt.xlabel('Shares 20+ Year Treasury Bond')
plt.show()

# Create Training and Tests
train = df[df.index < pd.to_datetime("2018-01-01", format='%Y-%m-%d')]
test = df[df.index >= pd.to_datetime("2018-01-01", format='%Y-%m-%d')]

plt.plot(train, color = "black")
plt.plot(test, color = "red")
plt.ylabel('iShares 20+')
plt.xlabel('Date')
plt.xticks(rotation=45)
plt.title("Train/Test graph")
plt.show()


from pmdarima import auto_arima
import warnings

warnings.filterwarnings('ignore')

#evaluate automatically which arima model is the best
data = df.set_index('Day')
train = df.iloc[:int(.8*(df.shape[0])),:]
auto_model = auto_arima(
  train,
  start_P=1,
  start_q=1,
  max_p=6,
  max_q=6,m=12,
  seasonal=True,
  max_P=2, 
  max_D=2,
  max_Q=2,
  max_d=2,
  trace=True,
  error_action='ignore',
  suppress_warnings=True,
  stepwise=True,
  information_criterion="aic",
  alpha=0.05,
  scoring='mse'
)


ARIMAmodel = ARIMA(df, order = (0, 1, 1))
model = ARIMAmodel.fit()
print(model.summary())

model.plot_diagnostics(figsize=(16,6))
plt.show()

from statsmodels.graphics.tsaplots import plot_predict
plot_predict(model)
plt.show()

#---------------------
plt.plot(train, color = "black", label = 'Training')
plt.plot(test, color = "red", label = 'Testing')
plt.ylabel('BTC Price')
plt.xlabel('Date')
plt.xticks(rotation=45)
plt.title("Train/Test split")
plt.show()


#Calculate Square error forectast
import numpy as np
from sklearn.metrics import mean_squared_error
arma_rmse = np.sqrt(mean_squared_error(test["BTC-USD"].values, y_pred_df["Predictions"]))
print("ARMA RMSE: ",arma_rmse)