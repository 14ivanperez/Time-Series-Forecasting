## Example of Time Series

### Introduction

More than 10 exercises where the use of Time Series knowledge is required. I srtart testing for the presence of unit roots, reviewing the auotocorrelation function, estimating models to produce forecasts and, finally, generating a Vector Autoregression model and a set impulse response functions.


### Installation and load data set

The file sales.dta contains 157 weekly observations on sales revenues (sales) and advertising
expenditure (adv) in millions of dollars for a large US department store for 2005-2007. The 
weeks are from December 28, 2004, to December 25, 2007.

```markdown
use sales.dta, // load the file sales.dta 

drop if sales==. // drop missing values: weekends, holidays etc 
rename t w //rename variable t as w (weeks) 

tsset w // tell stata to use weeks as the time index
```

### Topics discused

````markdown
# STATIONARITY
# TEST OF NON-STATIONARITY (DICKEY FULLER)
# AUTOCORRELATION FUNCTION
# INFORMATION CRITERION
# MODELS: AR, ARMA, VAR
````

### Exercises (graphics and images attached in index.md file)

#### (a) Graph the series for sales. Does it appear to be stationary, trend stationary or stochastic?
It seems to be stationary as there’s no trend, and the values are constantly going up and down 
around the mean, which seems to be approximately the same during different range periods.
It doesn’t neither seem to exist a high covariance between any value and its lagged ones.

#### (b) Formally test for the presence of a unit root. Explain what version of the test you used and why you used it.
I will use the Dickey–Fuller (ADF) test as it serves to test if the series follows a unit-root 
process, or in other words, if it’s non-stationary. The null hypothesis is that the variable 
contains a unit root(rho=1), and the alternative hypothesis is that the series is stationary/has 
no unit root process.
I’m using the ADF version with no trend because the autocorrelation graphs don’t seem to 
have a drift or linear trend as the data looks stationary and like a random walk. The variable 
tested is the log of sales revenues, as It is more appropriate than the actual values.
The coefficient of the regression is not closed to 1, and we can see that the Test Statistic is 
bigger than any critical value in absolute term, so we can reject the null hypothesis that there’s 
unit root in the series and we can confirm the time series is stationary.

#### (c) Examine the autocorrelation function and propose one or two possible models that may have generated the series. Where “c” is the difference of weekly log revenues of sales.
As it is also seen with the function corrgram, the ACF only has 1 lag out of the confidence 
interval and keeps oscillating while it is decaying. That suggests an AR(1) or MA(1). On the 
other hand, the PACF has several spikes and a big one at the end, which suggests the series is 
not random and that an ARMA model should be a used. Furthermore, the ACF is not decaying 
constantly, which means that just an AM model isn’t a good fit. To verify, if we test AR(1), 
MA(1) and ARMA(1,1) models, we appreciate the ARMA (1,1) has the most biggest BIC value in 
absolute terms.

#### d) Estimate the model(s) proposed in part (c).
I estimate all 3 models: First one is AR(1) whose rho is -0.0397 and significant.
The MA(1): Coefficient of 0.6237
The AR(1)MA(1) graph model is in index.md
 
#### (e) Assess your model(s) using if possible formal tests and/or an information criterion.
I will use the next information criterion to assess the arma model and select the lag length:
-AIC= ln(SSR(k)/T) + (k+1) 2/T
-BIC= ln(SSR(k)/T) + (k+1) lnT/T
After calculate different BIC with different lags in AR, I noticed the ARMA(1,1) model is the 
best one fitting as it has the most negative BIC.
Moreover, the ARMA(1,1) has a higher log likelihood than the other models.

#### (f) Estimate the model you deem most appropriate and estimate it again on the first 100  observations.
Now I use arima command with w<101 where w=weeks
The rhos are 0.4848 and -1 respectively. Only one coefficient is significant.

#### (g) Generate forecasts for the next 57 observations, and plot forecasted and actual 
observations. Assess the forecasting power of your model.
The forecast has less variance than the actual values. But it is a good forecast as It’s most of 
the time between the 95% level error bands.

#### (h) Estimate a VAR model of sales and advertising. Be sure to justify your choice of lag length. 
VAR suggests using 1 lag length, as it has the biggest values of HQIC and SBIC.
Then, this is our VAR model with lag 1. The R-squared is not too big as there’s no unit root.

#### (i) Do sales Granger cause advertising or vice-versa of both? Carefully state the null hypothesis of any tests you employ and carefully state the conclusion.
When we add more variables to a regression could be beneficial or not: As there are more 
variable, it is easier to track the data and reduce the error term (1st source of error), but on the 
other hand, we estimate each parameter less precise as we have more (2nd source of error). 
Granger is a formal test that helps us to guess if we should add more variables or not to the 
regression.
In this case, the probability that null hypothesis (advertising doesn’t explain sales revenues) is 
true equals 0, so we can reject the null hypothesis and say that advertising expenditure
granger causes sales revenues, which It makes sense from an economic and practical view.
On the other hand, the probability that sales revenues don’t explain advertising expenditure is 
much bigger than any possible critical value. Thus, sales revenues don’t granger cause 
advertising, which follows the economic intuition.

#### (j) Generate a set impulse response functions and discuss their economic interpretation.
As we can see, the impulse response function of sales revenues on advertising spending is 0 
during all the lags. It does make sense as in real businesses sales revenues have no effect on 
the quantity a firm spends on advertising. But the opposite is true: the intuition tells us that 
the more a firms spends on advertising, the more it will sell. In fact, that’s what the irf graph of 
advertising on sales shows us: Sales revenues are affected by the advertising expenditure from 
1 week ago, but not more periods ago.

#### (k) Generate a forecast of sales from the VAR and compare it to the univariate forecast of part (g).
Graph with actual data, forecast using VAR and forecast using AR from part g).
As It is appreciated, the forecast from the VAR follows very well the path of the actual data
````
___


### References

"Time Series Analysis: Forecasting and Control", 5th Edition, George E. P. Box, Gwilym M. Jenkins, Gregory C. Reinsel, Greta M. Ljung

Sources: dataset: file sales.dta contains 157 weekly observations on sales revenues (sales) and advertising expenditure (adv) in millions of dollars for a large US department store for 2005-2007
