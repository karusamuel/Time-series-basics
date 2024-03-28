# Time-Series-Data-Exploration and modeling

Time series data is a sequence of data points collected, recorded, or generated at equally spaced intervals over a period. Each data point in a time series is associated with a timestamp, indicating when it was observed or measured. This type of data is used to analyze trends, patterns, and behaviors that evolve over time. Time series data is prevalent in various fields like finance, economics, weather forecasting, stock market analysis, signal processing, and many others. Examples include daily stock prices, temperature recordings, weekly sales figures, hourly energy consumption, and monthly unemployment rates. Analyzing time series data often involves techniques like forecasting future values, identifying patterns, and understanding the underlying factors affecting the observed trends.

> if data is not equally spaced up_sample or down_samble to equal spacing

## Working with Time Series Data 

When working with Time Series data the following steps are helpful to make your Time series dataset easier to manipulate 



> Convert date/time colum to type  DateTime

```python
# e.g
# note format to be the format of the date time more on the 
# date time formats
# https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior
df['Date'] = pd.to_datetime(df["Date"],format='%d/%m/%y')
```

> set the date/time column as index
```python
df.set_index("Date",inplace=True)
df.info()
```


## Time Series Modeling 

### ARIMA

ARIMA stands for Autoregressive Integrated Moving Average. It's a popular and powerful statistical method used for analyzing and forecasting time series data. 

The Components of ARIMA are:

#### Autoregression (AR): (p)

This refers to a model that uses the dependent relationship between an observation and a number of lagged observations (past values in the series). An AR(p) model predicts the next value in the series based on a linear combination of the past p values.

> use the PACF plot to determine this value

#### Integrated (I): (d)

 This component represents the differencing of raw observations to make the series stationary. Stationarity is crucial in time series analysis because many forecasting methods assume that the time series is stationary (i.e., its statistical properties like mean and variance remain constant over time). The 'I' in ARIMA indicates how many differences are needed to achieve stationarity.

 > use differencing and adfuller to check for stationality  d is number of differencing to make time series stationary if already stationary then 0

#### Moving Average (MA): (q)
 This component models the relationship between an observation and residual errors from a moving average model applied to lagged observations. An MA(q) model predicts the next value in the series based on a linear combination of past error terms.

 > use the ACF plot to find this value

#### The ARIMA model is denoted as ARIMA(p, d, q), where:

    p is the order of the autoregressive part. (pacf)

    d is the degree of differencing needed to achieve stationarity.(differencing)

    q is the order of the moving average part. (acf)

### CONS

ARIMA assumes the the time Series is stationary and therefore must be made stationary before modeling

### Work Flow

``` python
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Generate or load your time series data
# For example, creating a synthetic time series
np.random.seed(0)
date_range = pd.date_range(start='2023-01-01', periods=100, freq='D')
values = np.random.randn(100).cumsum()  # Generating random data for demonstration
time_series = pd.Series(values, index=date_range)

# Plotting the time series data
time_series.plot(figsize=(10, 6))
plt.title('Sample Time Series Data')
plt.xlabel('Date')
plt.ylabel('Value')
plt.show()

# Check for stationarity or apply differencing if needed
# For example, to make the series stationary
stationary_series = time_series.diff().dropna()

# Plotting ACF and PACF to determine ARIMA parameters
plot_acf(stationary_series)
plt.title('Autocorrelation Function (ACF)')
plt.show()

plot_pacf(stationary_series)
plt.title('Partial Autocorrelation Function (PACF)')
plt.show()

# Fit ARIMA model
# For example, using ARIMA(1, 1, 1) for demonstration purposes
# replace  your own pdq values (pacf,difrencing,acf)
model = ARIMA(time_series, order=(1, 1, 1))
result = model.fit()

# Summary of the ARIMA model
print(result.summary())

# Forecast future values
# e.g steps = 10 will be
# 10 day if our data is in days 
# 10 month if our data is in months 
# 10 years if our data is in years 
#etc
forecast_steps = 10
forecast = result.forecast(steps=forecast_steps)

# Plotting the original series and forecasted values
plt.figure(figsize=(10, 6))
plt.plot(time_series, label='Original Series')
plt.plot(forecast, label='Forecast', color='red')
plt.title('ARIMA Forecast')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.show()

```

## SARIMA

SARIMA stands for Seasonal Autoregressive Integrated Moving Average. It's an extension of the ARIMA model that incorporates seasonality into the analysis of time series data. SARIMA models are specifically designed to handle seasonal patterns that occur at regular intervals within a time series.

Similar to ARIMA, SARIMA models consist of three main components:

    Autoregressive (AR), Integrated (I), and Moving Average (MA) Components: These components function similarly to those in ARIMA, capturing the non-seasonal aspects, trend, and stationarity in the time series data.
and

    Seasonal Component: This accounts for the seasonal patterns observed in the data.

   

SARIMA models are particularly useful for time series data that exhibit seasonal patterns, such as sales figures affected by seasonal trends, quarterly financial data, or monthly climate observations. They allow for the modeling of both non-seasonal and seasonal dynamics in the data, enabling better forecasting and capturing more  patterns compared to  ARIMA models.

``` python 

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Generate or load your time series data
# For example, creating a synthetic time series
np.random.seed(0)
date_range = pd.date_range(start='2023-01-01', periods=200, freq='D')
values = np.random.randn(200).cumsum()  # Generating random data for demonstration
time_series = pd.Series(values, index=date_range)

# Fit SARIMA model
# For example, using SARIMA(1, 1, 1)(1, 1, 1, 12) for demonstration purposes
# Same as ARIMA use pacf differencing and  acf  to get values
# s=seasonal_patten
# e.g 12 monthly data
#order = (p,d,q) ,seasonal_order = (p,d,q,s)

model = SARIMAX(time_series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
result = model.fit()

# Summary of the SARIMA model
print(result.summary())

# Forecast future values
forecast_steps = 10
forecast = result.get_forecast(steps=forecast_steps)

# Extracting forecasted values and confidence intervals
# Confidence intervals in a SARIMA model represent a range of values within which we can reasonably expect the true future observations to fall

forecast_values = forecast.predicted_mean
confidence_intervals = forecast.conf_int()

# Plotting the original series, forecasted values, and confidence intervals
plt.figure(figsize=(10, 6))
plt.plot(time_series, label='Original Series')
plt.plot(forecast_values.index, forecast_values.values, label='Forecast', color='red')
plt.fill_between(confidence_intervals.index, confidence_intervals.iloc[:, 0], confidence_intervals.iloc[:, 1], color='pink', alpha=0.3, label='Confidence Intervals')
plt.title('SARIMA Forecast')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.show(

```

## PMDARIMA( auto ARIMA)

> Best when modelling ARIMA and SARIMAX

> combines both ARIMA and SARIMA and automates pdq selection 

pmdarima aims to simplify the process of fitting ARIMA models by automating certain steps like identifying the optimal parameters (p, d, q) for ARIMA and SARIMA models.



```bash
# in your python enviroment
pip install pmdarima
```

[pmd_arima_Docs](https://pypi.org/project/pmdarima/)

Key features of pmdarima include:

**Automatic ARIMA:**

 It offers a function (auto_arima()) that automatically selects the best ARIMA or SARIMA model parameters based on various criteria, such as AIC (Akaike Information Criterion) or BIC (Bayesian Information Criterion).

**Seasonality Handling:**

 pmdarima supports modeling and forecasting of seasonal time series data, making it convenient for users dealing with seasonal patterns.

**Cross-Validation:** 

It allows users to perform cross-validation to assess the model's performance and generalization on unseen data.

``` python

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pmdarima import auto_arima

# Generate or load your time series data
# For example, creating a synthetic time series
np.random.seed(0)
date_range = pd.date_range(start='2023-01-01', periods=100, freq='D')
values = np.random.randn(100).cumsum()  # Generating random data for demonstration
time_series = pd.Series(values, index=date_range)

# Fit ARIMA model using auto_arima
# You can let auto_arima determine the best parameters
# set seasonal = true is data has seasonality 
# you can also manually set limits for (p,d,q,s)
# more here 
#https://pypi.org/project/pmdarima/
model = auto_arima(time_series, seasonal=False, trace=True)

# Summary of the ARIMA model
print(model.summary())

# Forecast future values
forecast_steps = 10
forecast = model.predict(n_periods=forecast_steps)

# Plotting the original series and forecasted values
plt.figure(figsize=(10, 6))
plt.plot(time_series, label='Original Series')
plt.plot(date_range[-1] + pd.to_timedelta(np.arange(1, forecast_steps + 1), unit='D'), forecast, label='Forecast', color='red')
plt.title('ARIMA Forecast using pmdarima')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.show()

```


## Facebook Prophet 

> more automation 

Facebook Prophet  is an open-source forecasting tool developed by Facebook's Core Data Science team. It's designed to simplify the process of time series forecasting and is particularly user-friendly for analysts and data scientists. 

Key features of Prophet include:

**Automatic Seasonality Detection:** 

Prophet can automatically detect various types of seasonalities in the data, including yearly, weekly, and daily patterns, as well as holidays that might affect the time series.

**Flexible Trend Modeling:**

It allows for both linear and non-linear trend modeling, providing flexibility to capture various trends present in the data.

**Holiday Effects:**

Prophet enables users to include holiday effects, allowing the model to consider the impact of holidays or specific events on the time series.

**Robustness to Missing Data and Outliers:**

Prophet is designed to handle missing data and outliers in a robust manner, minimizing their impact on the forecasting process.

**Scalability:**

It's capable of handling large datasets efficiently, making it suitable for use cases with substantial amounts of time series data.

**Interpretability and Tunability:** 

Prophet provides intuitive parameters for users to control the forecasting process, and the models generated are relatively interpretable.

``` python

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fbprophet import Prophet

# Generate or load your time series data
# For example, creating a synthetic time series
np.random.seed(0)
date_range = pd.date_range(start='2023-01-01', periods=100, freq='D')
values = np.random.randn(100).cumsum()  # Generating random data for demonstration
time_series = pd.DataFrame({'ds': date_range, 'y': values})

# Initialize and fit the Prophet model
model = Prophet()
model.fit(time_series)

# Create a dataframe for future predictions
future = model.make_future_dataframe(periods=10)  # Forecasting 10 additional periods

# Generate forecast
forecast = model.predict(future)

# Plotting the forecast
fig, ax = plt.subplots(figsize=(10, 6))
model.plot(forecast, ax=ax)
plt.title('Prophet Forecast')
plt.xlabel('Date')
plt.ylabel('Value')
plt.show()

```

## Other Models

Read on LSTM 

> ps they require a lot of data and a generally more complex to built tune and interpret compared to the above models 
