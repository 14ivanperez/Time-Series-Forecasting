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
import numpy as np

data = pd.DataFrame(investpy.get_etf_historical_data(etf='iShares 20+ Year Treasury Bond', country='united states', from_date='01/01/2012', to_date='01/01/2022'))
df = data['Close']

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


#Build Arima for train data
import itertools
warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')
p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))

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

#Produce Arima model
mod = sm.tsa.statespace.SARIMAX(train,
                                order=(1, 1, 1),
                                seasonal_order=(1, 1, 0, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
results = mod.fit()
print(results.summary().tables[1])

results.plot_diagnostics(figsize=(16,6))
plt.show()

#Predict using Arima
pred = results.get_prediction(start=pd.to_datetime('2018-01-01'), dynamic=False)
pred_ci = pred.conf_int()
ax = df['2012':].plot(label='observed')
pred.predicted_mean.plot(ax=ax, color = 'red', label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)
ax.set_xlabel('Date')
ax.set_ylabel('iShares 20+ Year Treasury Bond')
plt.legend()
plt.show()


#Calculate Square error forectast
import numpy as np
from sklearn.metrics import mean_squared_error
arma_rmse = np.sqrt(mean_squared_error(test["BTC-USD"].values, y_pred_df["Predictions"]))
print("ARMA RMSE: ",arma_rmse)
