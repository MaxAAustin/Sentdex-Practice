import pandas as pd
import quandl
import math
import numpy as np
from sklearn import preprocessing
import matplotlib as mp

'''
df is the data frame (what we are looking through)
quandl.get() retrieves stock info using the Quandl code String value passed to the method.
'''

df = quandl.get('WIKI/GOOGL')

df = df[['Adj. Open', 'Adj. High','Adj. Low', 'Adj. Close','Adj. Volume']]
df['HL_PCT'] = (df['Adj. High']-df['Adj. Close'])/df['Adj. Close'] * 100
df['PCT_Change'] = (df['Adj. Close']-df['Adj. Open'])/df['Adj. Open'] * 100

'''
defines a new data frame with only the values we want to see when we call the head method
choosing the features of the data to manipulate directly impacts the capability of a classifier.
'''

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
uses the Math lib to call ceil() and round numbers to the next whole number. ceil() returns a float, though. It can be difficult to work with floats. That's why we made it an int.
within ceil, we are projecting out 10% (0.1) of the data frame
'''
forecast_out = int(math.ceil(0.01*len(df)))

''' 
We needed the forecast_out to be an int because we are shifting the columns.
defining a label allows us to easily modify the feature we are trying to predict in the forecast_col variable later on
df[forecast_col].shift(-forecast_out) shifts the column up (because we set the shift as negative)
Each label for a column will be the 'Adj. Close' price for the future (because we set that as the value for the 
forecast_col variable). 
'''
df['label'] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)
print(df.head())