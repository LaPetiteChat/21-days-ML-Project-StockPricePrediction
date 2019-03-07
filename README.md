# 21-days-ML-Project-StockPricePrediction
21 days ML Project StockPricePrediciton

Prediction of Stock Prices

Step 1: Preparing data for analysis (EDA)

As this is a stock with clean data, all we have to do is to import the data and apply linear regression model for the stock.

First we examine the raw data’s pattern by looking at the correlation between the closing price and the data/time. 

As we can see, the stock price increases slowly with the time axis. 

Step 2: The codes for EDA is as follows:

#import pandas and read Excel with Pandas
import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile

df=pd.read_excel(‘stockdata.xlsx’, sheetname=’Sheet1’)

print(“Column headings:”)
print(df.columns)

#save an entire column into a list:
listDate=df['Date']
listCP=df['Close Price']

step 3: 
#用代码实现线性回归程序
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_sqaured_error, r2_score

Dates=np.array(['Date'])
Prices=np.array(df['Close Price'])

#将特征数据分为训练集和测试集，超参暂定为100天，其他都用于训练。
X_train=Dates[1:100]
X_train=X_train.reshape(-1, 1)

X_test=Dates[100:2019]
X_test=X_test.reshape(-1,1)

#把目标数据（特征对应的真实值）也分为训练集和测试集
y_train=Prices[1:100]	
y_train=y_train.reshape(-1,1)

y_train=Prices[100:2019]
y_train=y_train.reshape(-1,1)

#创建线性回归模型
regr=linear.model.Linear-Regression( )

  # 用训练集训练模型
    regr.fit(X_train, y_train)

    # 用训练得出的模型进行预测
    stock_y_pred = regr.predict(X_test)

    # 将测试结果以图标的方式显示出来
    plt.scatter(X_test, y_test,  color='black')
    plt.plot(X_test, stock_y_pred, color='blue', linewidth=3)

    plt.xticks(())
    plt.yticks(())
    plt.show()
