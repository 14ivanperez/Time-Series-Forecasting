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
train = df[df.index < pd.to_datetime("2020-11-01", format='%Y-%m-%d')]
test = df[df.index > pd.to_datetime("2020-11-01", format='%Y-%m-%d')]

plt.plot(train, color = "black")
plt.plot(test, color = "red")
plt.ylabel('iShares 20+')
plt.xlabel('Date')
plt.xticks(rotation=45)
plt.title("Train/Test graph)
plt.show()


#Build Arima for train data
mod = sm.tsa.arima.ARIMA(train, order=(1, 1, 2))
model_fit = mod.fit()
print(model_fit.summary())

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

#Produce Arima model
mod = sm.tsa.statespace.SARIMAX(df,
                                order=(1, 1, 1),
                                seasonal_order=(1, 1, 0, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
results = mod.fit()
print(results.summary().tables[1])

results.plot_diagnostics(figsize=(16,6))
plt.show()


arma_rmse = np.sqrt(mean_squared_error(test["BTC-USD"].values, y_pred_df["Predictions"]))
print("RMSE: ",arma_rmse)

# Make as pandas series
fc_series = pd.Series(fc, index=test.index)
lower_series = pd.Series(conf[:, 0], index=test.index)
upper_series = pd.Series(conf[:, 1], index=test.index)

# Graph Plot
plt.figure(figsize=(12,5), dpi=100)
plt.plot(train, label='training')
plt.plot(test, label='actual')
plt.plot(fc_series, label='forecast')
plt.fill_between(lower_series.index, lower_series, upper_series, 
                 color='k', alpha=.15)
plt.title('Forecast vs Actuals')
plt.legend(loc='upper left', fontsize=8)
plt.show()