#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import datetime


# In[2]:


df = pd.read_csv('all_stocks_5yr.csv')


# In[3]:


df.head()


# In[4]:


df.tail()


# In[5]:


df.shape


# In[6]:


df.info()


# In[7]:


df['date'] = pd.to_datetime(df["date"], format="%Y/%m/%d")


# In[8]:


df.info()


# In[9]:


stocks = list(df.Name.unique())


# In[10]:


type(stocks)


# In[11]:


#Number of stock available in the data
len(stocks)


# In[12]:


print(stocks)


# In[13]:


#lets pick one stock at this time and try to work around it


# In[14]:


#AAPl

df_aapl = df[df['Name']=='AAPL']


# In[15]:


df_aapl.head()


# In[16]:


df_aapl.info()


# In[17]:


#Description of apple stock prices
df_aapl.describe()


# In[18]:


fig, (axis1, axis2) = plt.subplots(2, 1, figsize=(18,15))

axis1.plot(df_aapl['date'],df_aapl['open'], color = 'black', label = 'open')
axis1.plot(df_aapl['date'],df_aapl['high'], color = 'green', label = 'high')
axis1.plot(df_aapl['date'],df_aapl['low'], color = 'red', label = 'low')
axis1.plot(df_aapl['date'],df_aapl['close'], color = 'blue', label = 'close')
axis1.set_xlabel('Dates')
axis1.set_ylabel('Stock Prices')
axis1.set_title('Stock prices history')
axis1.legend()

axis2.plot(df_aapl['date'],df_aapl['volume'], color = 'orange', label = 'volume')
axis2.set_xlabel('Dates')
axis2.set_ylabel('Stocks Volume')
axis2.set_title('Volume traded overtime')
axis2.legend()

plt.show()

There's a significant decline in stock prices is visible between 2015 and mid of 2016 but overall trend of stock prices is positive.
The volume of stocks traded has been decreased with time. Possibly due to high prices of the stocks.
# In[19]:


df_aapl.index


# In[20]:


#Reset Index
#df_aapl.reset_index()


# In[21]:


#Set date as index in dataframe
df_aapl = df_aapl.set_index('date', drop=True)


# In[22]:


df_aapl.head()


# In[23]:


#check if the datatype is correct
df_aapl.index


# In[24]:


#Insights by resampling
df_aapl.resample(rule = 'A').min()


# In[25]:


df_aapl.resample(rule = 'A').max()


# In[26]:


#Yearly insights
fig, ax = plt.subplots(figsize = (10,5))

ax.plot(df_aapl.resample(rule = 'A').min()['close'], color='red', label = 'Minimum Price')
ax.plot(df_aapl.resample(rule = 'A').max()['close'], color='blue', label = 'Maximum Price')
ax.set_xlabel('Dates')
ax.set_ylabel('Stock Prices')
ax.set_title('Minimum and Maximum closing prices -- YEARLY')
ax.legend()


plt.show()

The minimum and maximum prices per year follows the same pattern. The prices shoots up from 2017.
# In[27]:


df_aapl.resample(rule = 'Q').min()


# In[28]:


#Quarterly insights
fig, ax = plt.subplots(figsize = (14,5))

ax.plot(df_aapl.resample(rule = 'Q').min()['close'], color='red', label = 'Minimum Price')
ax.plot(df_aapl.resample(rule = 'Q').max()['close'], color='blue', label = 'Maximum Price')
ax.set_xlabel('Dates')
ax.set_ylabel('Stock Prices')
ax.set_title('Minimum and Maximum closing prices -- QUARTERLY')
ax.legend()


plt.show()

Prices started to drop from third quarter of 2015 to second quarter of 2016 and then the rise in prices can be seen.
# In[29]:


#Quarterly insights
fig, ax = plt.subplots(figsize = (15,5))

ax.plot(df_aapl.resample(rule = 'M').min()['close'], color='red', label = 'Minimum Price')
ax.plot(df_aapl.resample(rule = 'M').max()['close'], color='blue', label = 'Maximum Price')
ax.set_xlabel('Dates')
ax.set_ylabel('Stock Prices')
ax.set_title('Minimum and Maximum closing prices -- MONTHLY')
ax.legend()


plt.show()


# In[30]:


samp = df_aapl.resample(rule = 'M').min()
print(samp.index)


# In[31]:


#Closer insights between 2015 and 2017 insights
df_aapl_15_17 = df_aapl["2014-12-31":"2016-12-31"]

fig, ax = plt.subplots(figsize = (15,5))

ax.plot(df_aapl_15_17.resample(rule = 'M').min()['close'], color='red', label = 'Minimum Price')
ax.plot(df_aapl_15_17.resample(rule = 'M').max()['close'], color='blue', label = 'Maximum Price')
ax.set_xlabel('Dates')
ax.set_ylabel('Stock Prices')
ax.set_title('Minimum and Maximum closing prices -- Between 2015 and 2017')
ax.legend()


plt.show()

Prices precisely started to go down in 5th month of 2015 and then started to rise back up from 6th month of 2016.
# In[32]:


#Smoothing
df_aapl_ma = pd.DataFrame(df_aapl['close'].copy())


# In[33]:


type(df_aapl_ma)


# In[34]:


df_aapl_ma.head()


# In[35]:


df_aapl_ma['ma_5'] = df_aapl_ma['close'].rolling(5).mean()
df_aapl_ma['ma_10'] = df_aapl_ma['close'].rolling(10).mean()
df_aapl_ma['ma_15'] = df_aapl_ma['close'].rolling(15).mean()
df_aapl_ma['ma_20'] = df_aapl_ma['close'].rolling(20).mean()
df_aapl_ma['ma_30'] = df_aapl_ma['close'].rolling(30).mean()


# In[36]:


df_aapl_ma.head()


# In[37]:


fig, ax = plt.subplots(figsize = (16,5))

ax.plot(df_aapl_ma['close'], color='black', label = 'Actual Closing Price')
ax.plot(df_aapl_ma['ma_5'], color='blue', label = 'Moving Average by 5')
ax.plot(df_aapl_ma['ma_10'], color='red', label = 'Moving Average by 10')
ax.plot(df_aapl_ma['ma_15'], color='green', label = 'Moving Average by 15')
ax.plot(df_aapl_ma['ma_20'], color='orange', label = 'Moving Average by 20')
ax.plot(df_aapl_ma['ma_30'], color='magenta', label = 'Moving Average by 30')
ax.set_xlabel('Dates')
ax.set_ylabel('Stock Prices')
ax.set_xlim(datetime.date(2014,12,31),datetime.date(2017,12,31))
ax.set_title('Moving Averages of closing price')
ax.legend()


plt.show()

As we increase the window size of the rolling function, the prizes are getting more smoothened out along the duration.
# In[38]:


#Exponential weighted moving average (EWMA)
df_aapl_ma['ema_20'] = df_aapl_ma['close'].ewm(span=20).mean()
df_aapl_ma['ema_30'] = df_aapl_ma['close'].ewm(span=30).mean()
df_aapl_ma.head()


# In[39]:


fig, ax = plt.subplots(figsize = (16,5))

ax.plot(df_aapl_ma['close'], color='black', label = 'Actual Closing Price')
ax.plot(df_aapl_ma['ma_20'], color='magenta', label = 'Moving Average by 20')
ax.plot(df_aapl_ma['ema_20'], color='blue', label = 'Exponential Moving Average by 20')

ax.set_xlabel('Dates')
ax.set_ylabel('Stock Prices')
ax.set_xlim(datetime.date(2014,12,31),datetime.date(2017,12,31))
ax.set_title('Exponential Moving Averages of closing price')
ax.legend()


plt.show()


# In[40]:


fig, ax = plt.subplots(figsize = (16,5))

ax.plot(df_aapl_ma['close'], color='black', label = 'Actual Closing Price')
ax.plot(df_aapl_ma['ma_30'], color='magenta', label = 'Moving Average by 30')
ax.plot(df_aapl_ma['ema_30'], color='blue', label = 'Exponential Moving Average by 30')

ax.set_xlabel('Dates')
ax.set_ylabel('Stock Prices')
#ax.set_xlim(datetime.date(2014,12,31),datetime.date(2017,12,31))
ax.set_title('Exponential Moving Averages of closing price')
ax.legend()


plt.show()


# In[41]:


df_aapl_ma['ema_0.2'] = df_aapl_ma['close'].ewm(alpha = 0.2).mean()
df_aapl_ma['ema_0.6'] = df_aapl_ma['close'].ewm(alpha = 0.6).mean()
df_aapl_ma['ema_0.9'] = df_aapl_ma['close'].ewm(alpha = 0.9).mean()

df_aapl_ma.head()


# In[120]:


fig, ax = plt.subplots(figsize = (16,5))

ax.plot(df_aapl_ma['close'], color='black', label = 'Actual Closing Price')
ax.plot(df_aapl_ma['ema_0.2'], color='magenta', label = 'EMA with alpha 0.2')
ax.plot(df_aapl_ma['ema_0.6'], color='blue', label = 'EMA with alpha 0.4')
#ax.plot(df_aapl_ma['ema_0.9'], color='red', label = 'EMA with alpha 0.9')

ax.set_xlabel('Dates')
ax.set_ylabel('Stock Prices')
ax.set_xlim(datetime.date(2014,12,31),datetime.date(2016,12,31))
ax.set_title('Exponential Moving Averages of closing price')
ax.legend()


plt.show()

The lesser the alpha is, the more smooth the data gets. As the value of alpha increases, it takes almost the path of original data.From the performed EDA, we were able to derive the following insights from the Apple stock-
1. The stock prices contains white noise and follows a random walk.
2. Data shows the positive trend in the stock prices.
3. Not enough evidence found to gurantee the presence of seasonality in the stock closing prices.In order to perform the ARIMA, the series should be stationary.
# In[43]:


#Test for stationarity
from statsmodels.tsa.stattools import adfuller

I am conducting Augmented Dickey Fuller test to check if the closing price series is stationary or not.
H0: The series is non-stationary
H1: The series is stationary

if the P-value is less than significance level (0.05) then we would reject the null hypothesis.
# In[44]:


result = adfuller(df_aapl['close'])
print(f"ADF Statistic: {result[0]}")
print(f"P-value: {result[1]}")

Since the P-value is much larger than significance value, I cannot reject null hypothesis. This implies that my series is not stationary and I need perform differencing to a certain degree in order to make the series stationary.
# In[45]:


#Auto Correlation Function (ACF)

from statsmodels.graphics.tsaplots import plot_acf


# In[46]:


fig, (ax1,ax2) = plt.subplots(1,2, figsize=(16,4))

ax1.plot(df_aapl['close'])
ax1.set_title('Original')
plot_acf(df_aapl['close'], ax=ax2);


# In[47]:


#Differencing (d)

diff_1 = df_aapl['close'].diff().dropna()

fig, (ax1,ax2) = plt.subplots(1,2, figsize=(16,4))

ax1.plot(diff_1)
ax1.set_title('Differenced Once')
plot_acf(diff_1, ax=ax2);


# In[48]:


diff_2 = df_aapl['close'].diff().diff().dropna()

fig, (ax1,ax2) = plt.subplots(1,2, figsize=(16,4))

ax1.plot(diff_2)
ax1.set_title('Differenced Twice')
plot_acf(diff_2, ax=ax2);

It looks like Differencing (d) 1 is good enough to fit ARIMA. Let us confirm the same below using pmdarima package.
# In[49]:


from pmdarima.arima.utils import ndiffs


# In[50]:


ndiffs(df_aapl['close'], test="adf")

This confirms the value of differencing (d = 1) for implementing ARIMA.
# In[51]:


#Lets find out the optimal value of p for ARIMA model

from statsmodels.graphics.tsaplots import plot_pacf


# In[52]:


fig, (ax1,ax2) = plt.subplots(1,2, figsize=(16,4))

ax1.plot(diff_1)
ax1.set_title('Differenced Once')
plot_pacf(diff_1, ax=ax2);

From the above plot, I can take the value of P as 8 or 9.
# In[53]:


#Lets find out the optimal value of q for ARIMA model

fig, (ax1,ax2) = plt.subplots(1,2, figsize=(16,4))

ax1.plot(diff_1)
ax1.set_title('Differenced Once')
plot_acf(diff_1, ax=ax2);

From the above plot, I can take the value of Q as 2.
# In[54]:


#Fitting the ARIMA model

from statsmodels.tsa.arima.model import ARIMA


# In[55]:


model = ARIMA(df_aapl['close'], order=(9,1,2))
result_1 = model.fit()


# In[56]:


print(result_1.summary())


# In[57]:


model_2 = ARIMA(df_aapl['close'], order=(8,1,2))
result_2 = model_2.fit()
print(result_2.summary())


# In[58]:


#result_2.forecast()


# In[59]:


#plot the residuals

residual_1 = pd.DataFrame(result_1.resid)

fig, (ax1,ax2) = plt.subplots(1,2, figsize=(16,4))

ax1.plot(residual_1)
ax2.hist(residual_1, density=True)

plt.show()


# In[60]:


residual_2 = pd.DataFrame(result_2.resid)

fig, (ax1,ax2) = plt.subplots(1,2, figsize=(16,4))

ax1.plot(residual_2)
ax2.hist(residual_2, density=True)

plt.show()

Let us find out the best model through Auto ARIMA
# In[61]:


from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")


# In[62]:


stepwise_fit = auto_arima(df_aapl['close'], start_p=1, start_q=1,
                      test='adf',
                      max_p=9, max_q=2,
                      m=1,             
                      d=1,          
                      seasonal=False,   
                      start_P=0, 
                      D=None, 
                      trace=True,
                      error_action='ignore',  
                      suppress_warnings=True, 
                      stepwise=False)

Now that we have the order for best possible model, let's finalize the ARIMA model with the order and evaluate the results.
# In[63]:


order = stepwise_fit.get_params()['order']


# In[64]:


#Pre-processing of data - Train and Test split

n = int(len(df_aapl) * 0.8)
train = df_aapl['close'][:n]
test = df_aapl['close'][n:]


# In[79]:


print(df_aapl['close'].head())
print('*********************************************')
print(df_aapl['close'].tail())
print('*********************************************')


# In[84]:


print(train.head())
print('*********************************************')
print(test.tail())
print('*********************************************')


# In[65]:


print(len(train))
print(len(test))


# In[66]:


len(df_aapl) == len(train) + len(test)


# In[67]:


train.head()


# In[68]:


model = ARIMA(train.values, order = order)
result = model.fit()
result.summary()


# In[69]:


res = pd.Series(result.forecast(len(test)))


# In[70]:


result.forecast()


# In[71]:


type(res)


# In[72]:


res_df = res.to_frame(name='predictions')
res_df = res_df.reset_index(drop=True)


# In[73]:


test_df = test.to_frame(name='Actual')
test_df = test_df.reset_index()


# In[74]:


eval_df = pd.concat([test_df,res_df],axis=1)
eval_df = eval_df.set_index('date', drop=True)


# In[75]:


eval_df


# In[76]:


fig, (axis1, axis2) = plt.subplots(2, 1, figsize=(18,15))

axis1.plot(train, color = 'blue', label = 'Train')
axis1.plot(test, color = 'green', label = 'Test')
axis1.plot(eval_df.predictions, color = 'orange', label = 'Predicted', linestyle = '--')
axis1.set_xlabel('Dates')
axis1.set_ylabel('Stock Prices')
axis1.set_title('Stock Price prediction')
axis1.legend()

axis2.plot(test, color = 'green', label = 'Actual')
axis2.plot(eval_df.predictions, color = 'orange', label = 'Predicted', linestyle = '--')
axis2.set_xlabel('Dates')
axis2.set_ylabel('Stock Prices')
axis2.set_title('Stock Price prediction')
axis2.legend()

plt.show()


# In[103]:


'''start = len(train)
end = len(train) + len(test) - 1
pred = result.get_prediction(start=start, end=end)
print(pred.predicted_mean)'''


# In[108]:


#Evaluation of the model

test.values.mean()


# In[107]:


from math import sqrt
root_mse = sqrt(mean_squared_error(eval_df['Actual'],eval_df['predictions']))
print(root_mse)

Interpretation of the model: The error is much higher with the ARIMA model and it should be as less as possible. The more the error is closer to 0, the better is the model.