#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv('all_stocks_5yr.csv',parse_dates=True, index_col='date')


# In[3]:


df.head()


# In[4]:


type(df.index)


# In[5]:


df = df[df['Name']=='AAPL']


# In[6]:


df.head()


# In[7]:


df.info()


# In[9]:


df = df[['close']]


# In[10]:


df.describe()

Calculate Percent change

The reason for using the percent change instead of the actual prices is the benefit of normalization as we can measure all the variables in a comparable metric. Apart from that, returns have more statistical properties than actual prices like stationarity, as in most of the cases the prices cannot be stationary but we can have stationary returns.

A stationary time series is one where statistical properties such as mean, variance, standard deviation, correlation remains constant over time.
# In[11]:


df['returns'] = df.close.pct_change()


# In[12]:


df.head()


# In[13]:


df['log_returns'] = np.log(1 + df['returns'])


# In[14]:


df.head()


# In[21]:


plt.figure(figsize = (16,4))
plt.plot(df.log_returns)
plt.title('Log Returns of Closing Prices')
plt.xlabel('Dates')
plt.ylabel('Log Returns')
plt.show()

Our mean is constant throughout the time here.
# In[22]:


df.dropna(inplace = True)


# In[23]:


df.head()


# In[24]:


#Input variable
X = df[['close','log_returns']].values


# In[25]:


X


# In[26]:


from sklearn.preprocessing import MinMaxScaler


# In[27]:


scaler = MinMaxScaler(feature_range =(0,1)).fit(X)
X_scaled = scaler.transform(X)


# In[32]:


X_scaled[:5]


# In[29]:


#Output Variable
Y = [x[0] for x in X_scaled]


# In[31]:


Y[:5]


# In[34]:


#Train and test Split
split = int(len(X_scaled) * 0.8)
print(split)


# In[35]:


X_train = X_scaled[:split]
X_test = X_scaled[split:len(X_scaled)]
Y_train = Y[:split]
Y_test = Y[split:len(Y)] 


# In[40]:


assert len(X_train) == len(Y_train)
assert len(X_test) == len(Y_test)


# In[102]:


len(X_test)


# In[50]:


n = 3
Xtrain = list()
Ytrain = list()
Xtest = list()
Ytest = list()
for i in range(n,len(X_train)):
    Xtrain.append(X_train[i-n:i,:X_train.shape[1]])
    Ytrain.append(Y_train[i])
for i in range(n,len(X_test)):
    Xtest.append(X_test[i-n:i,:X_test.shape[1]])
    Ytest.append(Y_test[i])


# In[71]:


Xtrain , Ytrain = (np.array(Xtrain),np.array(Ytrain))
Xtrain = np.reshape(Xtrain, (Xtrain.shape[0], Xtrain.shape[1], Xtrain.shape[2]))

Xtest, Ytest = (np.array(Xtest),np.array(Ytest))
Xtest = np.reshape(Xtest, (Xtest.shape[0], Xtest.shape[1], Xtest.shape[2]))


# In[75]:


print(Xtrain.shape)
print(Ytrain.shape)
print('---'*20)
print(Xtest.shape)
print(Ytest.shape)


# In[81]:


#LSTM Model
from keras.models import Sequential
from keras.layers import LSTM, Dense


# In[82]:


model = Sequential()
model.add(LSTM(4, input_shape=(Xtrain.shape[1],Xtrain.shape[2])))
model.add(Dense(1))
model.compile(loss="mean_squared_error", optimizer="adam")
model.fit(Xtrain, Ytrain, epochs=100, validation_data=(Xtest, Ytest), batch_size = 16, verbose=1)


# In[83]:


model.summary()


# In[84]:


trainPredict = model.predict(Xtrain)
testPredict = model.predict(Xtest)


# In[103]:


len(testPredict)


# In[106]:


len(Ytest)


# In[114]:


trainPredict = np.c_[trainPredict, np.zeros(trainPredict.shape)]
testPredict = np.c_[testPredict, np.zeros(testPredict.shape)]


# In[116]:


trainPredict = scaler.inverse_transform(trainPredict)
trainPredict = [x[0] for x in trainPredict]

testPredict = scaler.inverse_transform(testPredict)
testPredict = [x[0] for x in testPredict]


# In[132]:


print(len(testPredict))
print(type(testPredict))


# In[118]:


df.close.tail()


# In[137]:


eval_df = pd.DataFrame(df['close'][-249:])
eval_df = eval_df.reset_index()


# In[138]:


pred_df = pd.Series(testPredict).to_frame(name='Predictions')
pred_df = pred_df.reset_index(drop=True)


# In[139]:


eval_df = pd.concat([eval_df,pred_df],axis=1)
eval_df = eval_df.set_index('date', drop=True)


# In[140]:


eval_df


# In[143]:


fig, (axis1, axis2) = plt.subplots(2, 1, figsize=(18,15))

axis1.plot(df['close'][:-249], color = 'blue', label = 'Train')
axis1.plot(df['close'][-249:], color = 'green', label = 'Test')
axis1.plot(eval_df['Predictions'], color = 'orange', label = 'Predicted', linestyle = '--')
axis1.set_xlabel('Dates')
axis1.set_ylabel('Stock Prices')
axis1.set_title('Stock Price prediction using LSTM')
axis1.legend()

axis2.plot(eval_df['close'], color = 'green', label = 'Actual')
axis2.plot(eval_df['Predictions'], color = 'orange', label = 'Predicted', linestyle = '--')
axis2.set_xlabel('Dates')
axis2.set_ylabel('Stock Prices')
axis2.legend()

plt.show()


# In[142]:


from sklearn.metrics import mean_squared_error
from math import sqrt
root_mse = sqrt(mean_squared_error(eval_df['close'],eval_df['Predictions']))
print(root_mse)

If we consider the rmse value, it is a fine model. But analysing the above plot, the prediction is quite accurate in the near future and error is increasing over time.
# In[ ]:




