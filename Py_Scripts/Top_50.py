#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df = pd.read_csv('all_stocks_5yr.csv',parse_dates=True, index_col='date')


# In[3]:


df.head()


# In[5]:


df = df[['Name','close']]


# In[21]:


df_grp = df.groupby('Name').mean()


# In[22]:


df_grp = df_grp.reset_index()


# In[23]:


df_grp = df_grp.sort_values(by=['close'],ascending = False)
df_grp = df_grp.reset_index(drop=True)


# In[83]:


df_grp.head()


# In[27]:


top_50 = list(df_grp['Name'][:50])


# In[80]:


top_50[:5]


# In[29]:


len(top_50)


# In[30]:


temp = [df, df_grp]


# In[36]:


top_50_dfnames = list()
for x in top_50:
    top_50_dfnames.append('df_'+x)


# In[37]:


top_50_dfnames[:5]


# In[53]:


#List of dataframes of top 50 stocks of S&P 500
top_50_dfs = list()
for i in range(50):
    top_50_dfnames[i] = df[df['Name']==top_50[i]]
    top_50_dfs.append(top_50_dfnames[i])


# In[54]:


len(top_50_dfs)


# In[69]:


type(top_50_dfs[2])


# In[74]:


from pmdarima import auto_arima
from math import sqrt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")


# In[89]:


#ARIMA Model

def ARIMA_model(df):
    stepwise_fit = auto_arima(df['close'], start_p=1, start_q=1,
                      test='adf',
                      max_p=9, max_q=3,
                      m=1,             
                      d=1, max_d=3,          
                      seasonal=False,   
                      start_P=0, 
                      D=None, 
                      trace=False,
                      error_action='ignore',  
                      suppress_warnings=True, 
                      stepwise=False)
    order = stepwise_fit.get_params()['order']
    n = int(len(df) * 0.8)
    train = df['close'][:n]
    test = df['close'][n:]
    model = ARIMA(train.values, order = order)
    result = model.fit()
    res = pd.Series(result.forecast(len(test)))
    res_df = res.to_frame(name='predictions')
    res_df = res_df.reset_index(drop=True)
    test_df = test.to_frame(name='Actual')
    test_df = test_df.reset_index()
    eval_df = pd.concat([test_df,res_df],axis=1)
    eval_df = eval_df.set_index('date', drop=True)
    root_mse = sqrt(mean_squared_error(eval_df['Actual'],eval_df['predictions']))
    
    return root_mse


# In[91]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# In[92]:


#Linear Regression
def Linear_model(df):
    def lagit(df, lags):
        names = list()
        for i in range(1,lags+1):
            df['lag_'+str(i)] = df['close'].shift(i)
            names.append('lag_'+str(i))
        df.dropna(inplace=True)    
        return names
    
    lagnames = lagit(df,5)
    X = np.array(df[lagnames])
    Y = np.array(df['close'])
    x_train,x_test,y_train,y_test = train_test_split(X,Y, test_size=0.2, shuffle=False)
    lr = LinearRegression().fit(x_train,y_train)
    pred = lr.predict(x_test)
    n = len(y_test)
    eval_df = pd.DataFrame(df['close'][-n:])
    eval_df = eval_df.reset_index()
    pred_df = pd.Series(pred).to_frame(name='Predictions')
    pred_df = pred_df.reset_index(drop=True)
    eval_df = pd.concat([eval_df,pred_df],axis=1)
    eval_df = eval_df.set_index('date', drop=True)
    root_mse = sqrt(mean_squared_error(eval_df['close'],eval_df['Predictions']))
    
    return root_mse


# In[97]:


from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense


# In[98]:


#LSTM
def LSTM_model(df):
    df['returns'] = df.close.pct_change()
    df['log_returns'] = np.log(1 + df['returns'])
    df.dropna(inplace = True)
    X = df[['close','log_returns']].values
    scaler = MinMaxScaler(feature_range =(0,1)).fit(X)
    X_scaled = scaler.transform(X)
    Y = [x[0] for x in X_scaled]
    split = int(len(X_scaled) * 0.8)
    X_train = X_scaled[:split]
    X_test = X_scaled[split:len(X_scaled)]
    Y_train = Y[:split]
    Y_test = Y[split:len(Y)] 
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
    Xtrain , Ytrain = (np.array(Xtrain),np.array(Ytrain))
    Xtrain = np.reshape(Xtrain, (Xtrain.shape[0], Xtrain.shape[1], Xtrain.shape[2]))

    Xtest, Ytest = (np.array(Xtest),np.array(Ytest))
    Xtest = np.reshape(Xtest, (Xtest.shape[0], Xtest.shape[1], Xtest.shape[2]))
    model = Sequential()
    model.add(LSTM(4, input_shape=(Xtrain.shape[1],Xtrain.shape[2])))
    model.add(Dense(1))
    model.compile(loss="mean_squared_error", optimizer="adam")
    model.fit(Xtrain, Ytrain, epochs=100, validation_data=(Xtest, Ytest), batch_size = 16, verbose=1)
    trainPredict = model.predict(Xtrain)
    testPredict = model.predict(Xtest)
    trainPredict = np.c_[trainPredict, np.zeros(trainPredict.shape)]
    testPredict = np.c_[testPredict, np.zeros(testPredict.shape)]
    trainPredict = scaler.inverse_transform(trainPredict)
    trainPredict = [x[0] for x in trainPredict]

    testPredict = scaler.inverse_transform(testPredict)
    testPredict = [x[0] for x in testPredict]
    z = len(Ytest)
    eval_df = pd.DataFrame(df['close'][-z:])
    eval_df = eval_df.reset_index()
    pred_df = pd.Series(testPredict).to_frame(name='Predictions')
    pred_df = pred_df.reset_index(drop=True)
    eval_df = pd.concat([eval_df,pred_df],axis=1)
    eval_df = eval_df.set_index('date', drop=True)
    root_mse = sqrt(mean_squared_error(eval_df['close'],eval_df['Predictions']))
    
    return root_mse


# In[99]:


temp = LSTM_model(top_50_dfs[2])


# In[100]:


temp


# In[101]:


ARIMA_error = list()
LR_error = list()

for x in top_50_dfs:
    ar_err = ARIMA_model(x)
    ARIMA_error.append(ar_err)
    lr_err = Linear_model(x)
    LR_error.append(lr_err)


# In[106]:


LSTM_error = list()

for x in top_50_dfs:
    lstm_err = LSTM_model(x)
    LSTM_error.append(lstm_err)


# In[107]:


len(LSTM_error)


# In[108]:


final_result = pd.DataFrame({'Ticks':top_50,
                            'ARIMA_error':ARIMA_error,
                            'LR_error':LR_error,
                            'LSTM_error':LSTM_error})


# In[109]:


final_result.head()


# In[110]:


final_result.to_csv('top_50.csv', index=False)


# In[ ]:




