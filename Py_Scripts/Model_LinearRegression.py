#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df = pd.read_csv('all_stocks_5yr.csv',parse_dates=True, index_col='date')


# In[3]:


df.head()


# In[4]:


type(df.index)


# In[9]:


#Stock Selection
df = df[df['Name']=='AAPL']


# In[10]:


df.info()


# In[11]:


df.describe()


# In[14]:


df = df[['close']]


# In[16]:


df.head()


# In[27]:


def lagit(df, lags):
    names = list()
    for i in range(1,lags+1):
        df['lag_'+str(i)] = df['close'].shift(i)
        names.append('lag_'+str(i))
    df.dropna(inplace=True)    
    return names


# In[28]:


lagnames = lagit(df,5)


# In[29]:


lagnames


# In[30]:


df.head()


# In[37]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# In[32]:


X = np.array(df[lagnames])


# In[35]:


Y = np.array(df['close'])


# In[40]:


#Preprocessing Data
x_train,x_test,y_train,y_test = train_test_split(X,Y, test_size=0.2, shuffle=False)


# In[52]:


print(len(x_train))
print(len(x_test))


# In[41]:


#Model creation and training
lr = LinearRegression().fit(x_train,y_train)


# In[46]:


pred = lr.predict(x_test)
print(pred.shape)


# In[76]:


len(y_test)


# In[49]:


type(pred)


# In[64]:


eval_df = pd.DataFrame(df['close'][-251:])
eval_df = eval_df.reset_index()


# In[68]:


pred_df = pd.Series(pred).to_frame(name='Predictions')
pred_df = pred_df.reset_index(drop=True)


# In[70]:


eval_df = pd.concat([eval_df,pred_df],axis=1)
eval_df = eval_df.set_index('date', drop=True)


# In[71]:


eval_df


# In[51]:


import matplotlib.pyplot as plt


# In[73]:


fig, (axis1, axis2) = plt.subplots(2, 1, figsize=(18,15))

axis1.plot(df['close'][:-252], color = 'blue', label = 'Train')
axis1.plot(df['close'][-251:], color = 'green', label = 'Test')
axis1.plot(eval_df['Predictions'], color = 'orange', label = 'Predicted', linestyle = '--')
axis1.set_xlabel('Dates')
axis1.set_ylabel('Stock Prices')
axis1.set_title('Stock Price prediction')
axis1.legend()

axis2.plot(eval_df['close'], color = 'green', label = 'Actual')
axis2.plot(eval_df['Predictions'], color = 'orange', label = 'Predicted', linestyle = '--')
axis2.set_xlabel('Dates')
axis2.set_ylabel('Stock Prices')
axis2.set_title('Stock Price prediction')
axis2.legend()

plt.show()


# In[74]:


from sklearn.metrics import mean_squared_error
from math import sqrt
root_mse = sqrt(mean_squared_error(eval_df['close'],eval_df['Predictions']))
print(root_mse)

The error is comparitively very low which indicates that the model is good.