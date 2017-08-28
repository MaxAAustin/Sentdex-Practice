import pandas as pd
import quandl
import math
import numpy as np
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression

'''
sklearn preprocessing allows us to perform scaling on the features of our data, which helps with processing speed, 
accuracy, etc. 
cross-validation allows us to avoid overfitting and create good training and testing samples. Saves time, 
shuffles data samples to reduce biases. 
numpy is a library that allows us to perform calculations, use arrays (Python has dictionaries, not arrays)
'''

df = quandl.get('WIKI/GOOGL')

df = df[['Adj. Open', 'Adj. High','Adj. Low', 'Adj. Close','Adj. Volume']]
df['HL_PCT'] = (df['Adj. High']-df['Adj. Close'])/df['Adj. Close'] * 100
df['PCT_Change'] = (df['Adj. Close']-df['Adj. Open'])/df['Adj. Open'] * 100


df = df[['Adj. Close','HL_PCT','PCT_Change','Adj. Volume']]

'''
forecast column can be set to whatever you want to look at in the future
df.fillna() method is used to fill not available data (na = missing) [in Pandas, na is NaN (not a #)]
with a specific value set. In this practice case it is -99999. 
You can't work with NaN data and have to replace it with something (e.g. fillna()) better than getting rid of a column.
'''

forecast_col = 'Adj. Close'
df.fillna(-99999, inplace=True)

'''
uses the Math lib to call ceil() and round numbers to the next whole number. ceil() returns a float, though. 
It can be difficult to work with floats. That's why we made it an int.
within ceil, we are projecting out 10% (0.1) of the data frame
'''
forecast_out = int(math.ceil(0.01*len(df)))
#outputs number of days in advance we are predicting
print(forecast_out)
df['label'] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)

"""
Features are defined by lowercase x and y.
The preprocessing.scale() is called on x to standardize the dataset along the x axis before sending it to the 
classifier. 
Preprocessing step adds to processing time. Not suitable for High Frequency Trading
"""
x = np.array(df.drop(['label'],1))
y = np.array(df['label'])
x = preprocessing.scale(x)
# ensures we only have x's where we have values for y because we shifted the forecasted values
y = np.array(df['label'])

"""
Creates our training and test sets and cross-validates them. 
THE CROSS-VALIDATION MODULE WAS DEPRECATED IN V0.18. IT NOW RESIDES IN THE MODEL_SELECTION MODULE. 
"""
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

'''
Shows how easy it is to switch out learning algorithm.
kernel is a similarity function you provide to the ML algorithm. Polynomial is the selected example. 
'''
clf2 = svm.SVR(kernel='poly')
clf2.fit(x_train, y_train)
accuracySVM = clf2.score(x_test, y_test)

'''
score() is synonymous with test
Used to separate data trained and tested on to prevent classifier from just recalling.
For Linear Regression, accuracy is the squared error. (R^2) 
'''
accuracy = clf.score(x_test,y_test)

print(accuracy)
print(accuracySVM)