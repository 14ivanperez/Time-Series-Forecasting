import yfinance as yf
import investpy            
import yahoo_fin.stock_info as si
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import numpy as np

data = pd.DataFrame(investpy.get_etf_historical_data(etf='iShares 20+ Year Treasury Bond', country='united states', from_date='01/01/2012', to_date='01/01/2022'))
df = data['Close']
df.sort_index(inplace=True)
df.plot()
plt.show()

import warnings
fig, ax = plt.subplots(figsize=(20, 6))
ax.plot(df,marker='.', linestyle='-', linewidth=0.5, label='Daily')
ax.plot(df.resample('1m').mean(),marker='o', markersize=8, linestyle='-', label='Monthly Mean Resample')
ax.set_ylabel('Prices')
ax.legend()
plt.show()


#test non-stationarity
from statsmodels.tsa.stattools import adfuller
from numpy import log
result = adfuller(df.dropna())
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])

sm.graphics.tsa.plot_acf(df.values.squeeze(), lags=100)
sm.graphics.tsa.plot_pacf(df.values.squeeze(), lags=100)
plt.show()

mod = sm.tsa.arima.ARIMA(df, order=(1, 1, 2))
model_fit = mod.fit()
print(model_fit.summary())

plot_predict(model_fit, dynamic=False)
plt.show()

from statsmodels.tsa.stattools import acf

def forecast(model,predict_steps,df):
    
    pred_uc = model.get_forecast(steps=predict_steps)

    #SARIMAXResults.conf_int, can change alpha,the default alpha = .05 returns a 95% confidence interval.
    pred_ci = pred_uc.conf_int()

    ax = df.plot(label='observed', figsize=(14, 7))
#     print(pred_uc.predicted_mean)
    pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
    ax.fill_between(pred_ci.index,
                    pred_ci.iloc[:, 0],
                    pred_ci.iloc[:, 1], color='k', alpha=.25)
    ax.set_xlabel('Date')

    plt.legend()
    plt.show()
    
       # Produce the forcasted tables 
    pm = pred_uc.predicted_mean.reset_index()
    pm.columns = ['Date','Predicted_Mean']
    pci = pred_ci.reset_index()
    pci.columns = ['Date','Lower Bound','Upper Bound']
    final_table = pm.join(pci.set_index('Date'), on='Date')
    
    return (final_table)
    

# Create Training and Tests
train = df.iloc[:1421]
train
test = df.iloc[1421:]
test

# Build Model with specific parameters
# model = ARIMA(train, order=(3,2,1))  
model = sm.tsa.arima.ARIMA(train, order=(1, 1, 1))  
fitted = model.fit()  

# Forecast
fc = fitted.forecast(15, alpha=0.05)  # 95% conf

# Make as pandas series
fc_series = pd.Series(fc, index=test.index)
lower_series = pd.Series(conf[:, 0], index=test.index)
upper_series = pd.Series(conf[:, 1], index=test.index)

# Plot
plt.figure(figsize=(12,5), dpi=100)
plt.plot(train, label='training')
plt.plot(test, label='actual')
plt.plot(fc_series, label='forecast')
plt.fill_between(lower_series.index, lower_series, upper_series, 
                 color='k', alpha=.15)
plt.title('Forecast vs Actuals')
plt.legend(loc='upper left', fontsize=8)
plt.show()