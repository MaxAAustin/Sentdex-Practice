import quandl, math, datetime
import numpy as np
from sklearn import preprocessing, model_selection, svm
import matplotlib.pyplot as plt
import pickle
from matplotlib import style

df = quandl.get('WIKI/GOOGL')

df = df[['Adj. Open', 'Adj. High','Adj. Low', 'Adj. Close','Adj. Volume']]
df['HL_PCT'] = (df['Adj. High']-df['Adj. Close'])/df['Adj. Close'] * 100
df['PCT_Change'] = (df['Adj. Close']-df['Adj. Open'])/df['Adj. Open'] * 100


df = df[['Adj. Close','HL_PCT','PCT_Change','Adj. Volume']]

forecast_col = 'Adj. Close'
df.fillna(-99999, inplace=True)

forecast_out = int(math.ceil(0.01*len(df)))
#outputs number of days in advance we are predicting
print(forecast_out)

df['label'] = df[forecast_col].shift(-forecast_out)


"""
Features are defined by lowercase x and y.
The preprocessing.scale() is called on x to standardize the dataset along the x axis before sending it to the 
classifier. 
Preprocessing step adds to processing time. Not suitable for High Frequency Trading
"""
x = np.array(df.drop(['label'],1)
x = x[:-forecast_out]
'''
x_lately is what we're predicting against. 
'''
x_lately = x[-forecast_out:]
x = preprocessing.scale(x)

df.dropna(inplace=True)
y = np.array(df['label'])
y = np.array(df['label'])

x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.2)

'''
Creates a classifier called clf that will use Linear Regression
fit() is synonymous with train. 
Used to pass in features and labels to fit/ train the classifier
n_jobs = -1 runs as many jobs (threads) as possible by your processor to improve efficiency/ performance on training 
data
'''

clf = LinearRegression(n_jobs=-1)
clf.fit(x_train,y_train)

accuracy = clf.score(x_test,y_test)

print(accuracy)
print(accuracySVM)