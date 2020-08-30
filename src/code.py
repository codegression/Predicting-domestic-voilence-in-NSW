#!/usr/bin/env python
# coding: utf-8

# # Predicting domestic violence in NSW

# Python code to analyse and forecast domestic violence in NSW based on data from the NSW Bureau of Crime Statistics and Research.

# # Loading libraries

# Let's load relevant Python libraries.

# In[300]:


import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
import matplotlib as mpl
import pmdarima as pm
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
import time
import warnings
warnings.filterwarnings("ignore")


# # Loading data

# The dataset was acquired from https://www.bocsar.nsw.gov.au/Pages/bocsar_datasets/Datasets-.aspxNSW. It contains monthly data on all criminal incidents recorded by police from 1995 to Mar 2020.

# In[2]:


data = pd.read_csv('Incident_by_NSW.csv')


# In[3]:


data.head()


# # Cleaning data

# Let's view all the columns to better understand the dataset.

# In[4]:


data.columns


# In[5]:


data.describe()


# In[6]:


data.info()


# Let's explore 'Offence category' and 'Subcategory'

# In[7]:


diagram = sns.countplot(x='Offence category', data=data)
diagram.set_xticklabels(diagram.get_xticklabels(), rotation=40, ha="right")


# In[8]:


data.loc[:,'Offence category'].value_counts(dropna=False)


# In[9]:


data.loc[:,'Subcategory'].value_counts(dropna =False)


# 'Subcategory' has 13 missing values. Lets look at them

# In[10]:


print(data)


# In[11]:


data[data['Subcategory'].isna()]


# If there is no subcategory, the value should be taken from 'Offence category'

# In[12]:


data.Subcategory.fillna(data['Offence category'], inplace=True)


# In[13]:


print(data)


# In[14]:


data.Subcategory.replace("Murder *", "Murder", inplace=True)
data.Subcategory.replace("Manslaughter *", "Manslaughter", inplace=True)
data.head()


# We can see that the first 5 columns (except 'Subcategory') and the last 5 columns are not necessary. We can disgard them.

# In[15]:


data.drop(data.iloc[:, 0:2], inplace=True, axis=1) #drop the 1st and the 2nd columns


# In[16]:


data.drop(data.iloc[:, 1:3], inplace=True, axis=1) #drop third to 5th columns


# In[17]:


data.drop(data.iloc[:, -5:], inplace=True, axis=1) #drop the last 5 columns


# In[18]:


data.head()


# Let's remove the column name 'Subcategory' because we want to convert it into an index.

# In[19]:


data.rename(columns={"Subcategory": ""}, inplace=True)


# In[20]:


data.head()


# In[21]:


data.set_index('',inplace=True)


# In[22]:


data.head()


# # Converting into time series

# Columns and rows need to be interchanged (transposed) because every row needs to be a time stamp.

# In[23]:


data = data.T #transpose dataframe


# In[24]:


data.head()


# The index column right now has string values which need to be converted into dates.

# In[26]:


data.index = pd.to_datetime(data.index, infer_datetime_format=True)


# Save the dataframe for future use.

# In[27]:


data.to_csv('crimes_nsw.csv')


# # Exploratory data analysis

# Let's give a variable name.

# In[29]:


domestic = data['Domestic violence related assault']


# In[174]:


domestic.plot(kind = 'line', color = 'red',label = 'Domestic violence',linewidth=2,alpha = 1,grid = True,linestyle = '-')
plt.legend(loc='upper right')     
plt.xlabel('')              
plt.ylabel('Number of cases')
plt.title('Monthly plot of domestic violence in NSW')            
plt.show()


# We can see in the above diagram that domestic violence in NSW follows a predictable pattern. Generally, there is an upward trend. The time series has two components: trend component and seasonal component. Let's explore more.

# In[173]:


domestic.resample('Q').mean().plot(kind = 'line', color = 'red',label = 'Domestic violence',linewidth=2,alpha = 1,grid = True,linestyle = '-')
plt.legend(loc='upper right')     
plt.xlabel('')              
plt.ylabel('Number of cases')
plt.title('Quarterly plot of domestic violence in NSW')            
plt.show()


# In[172]:


yearly = domestic.resample('Y').mean().to_frame()
y = np.array(yearly[yearly.columns[0]])
x = np.array(range(len(x))) 
model = np.poly1d(np.polyfit(x, y, 1))
yearly.insert(1, "Fitted", model(x), True) 

yearly[yearly.columns[0]].plot(kind = 'line', color = 'red',label = 'Domestic violence',linewidth=2,alpha = 1,grid = True,linestyle = '-')
yearly[yearly.columns[1]].plot(kind = 'line', color = 'blue',label = 'Linear trend',linewidth=2,alpha = 1,grid = True,linestyle = '-')
plt.legend(loc='upper right')     
plt.xlabel('')              
plt.ylabel('Number of cases')
plt.title('Yearly plot of domestic violence in NSW')            
plt.show()


# We can clearly see an upward trend on a yearly basis.

# In[206]:


yearly = domestic.to_frame()
yearly['Year'] = [d.year for d in yearly.index]
sns.set(rc={'figure.figsize':(15,8)})
sns.set(font_scale=1.2)
sns.boxplot(x='Year', y='Domestic violence related assault', data=yearly)


# ### Seasonal analysis

# In[161]:


seasonal = domestic.to_frame()
seasonal['Year'] = [d.year for d in seasonal.index]
seasonal['Month'] = [d.strftime('%b') for d in seasonal.index]
years = seasonal['Year'].unique()


# In[162]:


seasonal.head()


# In[155]:


np.random.seed(53)
colors = np.random.choice(list(mpl.colors.XKCD_COLORS.keys()), len(years), replace=False)


# In[176]:


for i, year in enumerate(years):
    y_data = seasonal.loc[seasonal.Year==year, :]
    y_data.set_index('Month', inplace=True, drop=True)
    y_data.drop(y_data.iloc[:, 1:2], inplace=True, axis=1)
    y_data['Domestic violence related assault'].plot(kind = 'line', color=colors[i],label = year,linewidth=2,alpha = 1,grid = True,linestyle = '-')
plt.legend(bbox_to_anchor=(1.05, 1))     
plt.xlabel('')              
plt.ylabel('Number of cases')
plt.title('Seasonal plot of domestic violence in NSW')            
plt.show()    


# As depicted in the above figure, domenstic violence peaks in December and January each year (holiday season). This may be due to more idle time and more drinking. Case numbers also rise in March.

# In[208]:


yearly = domestic.to_frame()
yearly['Year'] = [d.year for d in yearly.index]
yearly['Month'] = [d.strftime('%b') for d in yearly.index]
sns.set(rc={'figure.figsize':(15,8)})
sns.set(font_scale=1.2)
sns.boxplot(x='Month', y='Domestic violence related assault', data=yearly.loc[~yearly.Year.isin([1995, 2020]), :])


# July is the month with the lowest variance, coinciding with beginnings of financial years.

# # Time series forecasting

# Now let's start forecasting. The time series is obviously not stationary as evident from the figure below.

# In[215]:


sns.reset_orig()
plt.rcdefaults()
domestic.plot(kind = 'line', color = 'red',label = 'Domestic violence',linewidth=2,alpha = 1,grid = True,linestyle = '-')
plt.legend(loc='upper right')     
plt.xlabel('')              
plt.ylabel('Number of cases')
plt.title('Monthly plot of domestic violence in NSW')            
plt.show()


# There are two ways to perform forecasting. The fist way is to decompose the time series (detrend and remove seasonal components) and then perform forecasting using models such as ARMA. The second way involves using models that account for trends and seasonality. An example is SARIMA.

# In[247]:


decomposed = seasonal_decompose(domestic, extrapolate_trend='freq')
decomposed.plot()
plt.show()


# Decomposing the time series into three components makes it difficult to compose back in the forecasting stage. Let's just use Seasonal Autoregressive Integrated Moving Average (SARIMA). 
# 
# SARIMA has 6 parameters <font color=green>SARIMA Model (p, d, q) (P, D, Q, S)</font>
# (p, d, q) are exactly the same as that of ARIMA. These are non-seasonal parameters.
# (P, D, Q) are seasonal parameters and S is the period of seasonality. 
# 
# We can guess optimal values of the parameters from ACF and PACF data.
# 

# ### 0th Order Differencing

# In[264]:


x = plot_acf(domestic, lags=100)
x = plot_pacf(domestic, lags=50)


# ### 1st Order Differencing

# In[265]:


x = plot_acf(domestic.diff()[1:], lags=50)
x = plot_pacf(domestic.diff()[1:], lags=50)


# ### 2nd Order Differencing

# In[266]:


x = plot_acf(domestic.diff().diff()[2:], lags=50)
x = plot_pacf(domestic.diff().diff()[2:], lags=50)


# Since we can't conclusively derive the optimal values from the above plots, let's use a grid search.

# ## Splitting dataset

# In[267]:


len(domestic)


# The time series has 303 samples. Let's split into training and test sets.
# 1. Training <font color=green>The first 291 samples</font> (From January 1995 to March 2019)
# 3. Test set <font color=green>The final 12 samples </font> (From April 2019 to March 2020)

# In[294]:


trainingdata = domestic[:291]
testdata = domestic[291:]


# ### Optimization

# In[298]:


optimization = pm.auto_arima(trainingdata, 
                             start_p=1, max_p=24,
                             start_q=1, max_q=24,   
                             start_d=1, max_d=24, 
                             start_P=0, max_P=24,
                             start_Q=0, max_Q=24,
                             start_D=1, max_D=24, 
                             m=12,                                                      
                             seasonal=True,
                             trace=True,
                             error_action='ignore',  
                             suppress_warnings=True, 
                             stepwise=True)


# In[299]:


optimization.to_dict()


# The optimal values for (p, q, d)(P, Q, D, S) are (2, 1, 0) (1, 0, 1, 12) respectively.

# In[313]:


model = sm.tsa.statespace.SARIMAX(endog=trainingdata,
                                  order=(2,1,0),
                                  seasonal_order=(1,0,1,12),
                                  trend='c')


# In[314]:


residue = model.fit(disp=False)
print(residue.summary())


# ### In-sample testing

# In[329]:


prediction = residue.predict()
trainingdata.plot(kind = 'line', color = 'red',label = 'Actual',linewidth=2,alpha = 1,grid = True,linestyle = '-')
prediction.plot(kind = 'line', color = 'blue',label = 'Prediction',linewidth=2,alpha = 1,grid = True,linestyle = '-')
plt.legend(loc='upper right')     
plt.xlabel('')              
plt.ylabel('Number of cases')
plt.title('')            
plt.show()


# It looks good. It's zoom into last 48 months of the training data to see more clearly.

# In[338]:


trainingdata[-48:].plot(kind = 'line', color = 'red',label = 'Actual',linewidth=2,alpha = 1,grid = True,linestyle = '-')
prediction[-48:].plot(kind = 'line', color = 'blue',label = 'Prediction',linewidth=2,alpha = 1,grid = True,linestyle = '-')
plt.legend(loc='upper right')     
plt.xlabel('')              
plt.ylabel('Number of cases')
plt.title('In-sample testing')            
plt.show()


# The in-sample performance looks great.

# ### Out-of-sample testing

# The model was trained with samples from January 1995 to March 2019. Let's perform out-of-sample forecasting.
# Let's make the model predict the next 12 samples (From April 2019 to March 2020) and compare with the test data (which were not included in the training stage)
# 

# In[335]:


predictions = residue.forecast(12)
print(predictions)


# In[334]:


testdata


# The numbers look very similar. Let's plot them.

# In[337]:


testdata.plot(kind = 'line', color = 'red',label = 'Actual',linewidth=2,alpha = 1,grid = True,linestyle = '-')
predictions.plot(kind = 'line', color = 'blue',label = 'Prediction',linewidth=2,alpha = 1,grid = True,linestyle = '-')
plt.legend(loc='upper right')     
plt.xlabel('')              
plt.ylabel('Number of cases')
plt.title('Out-of-sample testing')            
plt.show()


# In[ ]:


The predictions are remarkably accurate!


# # Out-of-dataset prediction (Foreseeable future)

# Let's retrain with all the data (303 samples) and let the model perform forecasting for the next 5 years.

# In[349]:


final_model = sm.tsa.statespace.SARIMAX(endog=domestic,
                                  order=(2,1,0),
                                  seasonal_order=(1,0,1,12),
                                  trend='c')

final_residue = final_model.fit(disp=False)


# In[352]:


final_predictions = final_residue.forecast(60) #5 years = 60  months
print(final_predictions)


# In[353]:


domestic.plot(kind = 'line', color = 'red',label = 'Current data',linewidth=2,alpha = 1,grid = True,linestyle = '-')
final_predictions.plot(kind = 'line', color = 'blue',label = 'Forecast',linewidth=2,alpha = 1,grid = True,linestyle = '-')
plt.legend(loc='upper right')     
plt.xlabel('')              
plt.ylabel('Number of cases')
plt.title('Out-of-dataset prediction')            
plt.show()


# # Conclusion

# As depicted in the figure above, domestic violence in NSW will most likely continue to rise in the next 5 years unless the NSW government pays attention to it and takes aggressive countermeasures.
