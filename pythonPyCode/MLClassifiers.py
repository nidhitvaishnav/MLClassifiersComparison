
# coding: utf-8

# # Data Pre-processing

# #### reading data in the inputDataFrame
# providing file path of car dataset and reading those data in the Pandas dataframe,
# here, if there are null values, they will be converted into '?",
# we have provided header name list

# In[179]:


import pandas as pd
inputFilePath = 'https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data'
inputDataFrame = pd.read_csv(filepath_or_buffer = inputFilePath, na_values='?', skipinitialspace=True, 
                             names  = ['buying', 'maint', 'doors','persons','lug_boot','safety', 'class'])


# In[180]:


inputDataFrame.head()


# #### removing null values

# In[181]:


nullRemovedDataFrame = inputDataFrame.dropna()


# #### convert categorical data into numerical data

# In[182]:


#creating a data frame which contains the object type values
objDataFrame = nullRemovedDataFrame.select_dtypes(include=['object']).copy()
nRows, nCols = objDataFrame.shape
#for all column converting object type in categoty type and assigning
#appropriate code
for myIndex in range(0,nCols):
    headerName = objDataFrame.columns[myIndex]
    objDataFrame[headerName] = objDataFrame[headerName].astype("category")
    objDataFrame[headerName] = objDataFrame[headerName].cat.codes
    #writing objectDataFrame column to its respective dataFrame column
    nullRemovedDataFrame[headerName] = objDataFrame[headerName]            
#for myIndex -ends
numericDataFrame = nullRemovedDataFrame


# In[183]:


numericDataFrame.head()


# In[184]:


y = numericDataFrame[['class']]


# In[185]:


y.head()


# #### Scale the data:      
# scaling data using MinMaxScaler

# In[186]:


import numpy as np
#finding shape of dataframe
nRows, nCols = numericDataFrame.shape

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
numericDataFrame[numericDataFrame.columns] = scaler.fit_transform(numericDataFrame[numericDataFrame.columns])
scaledDataFrame = numericDataFrame


# In[187]:


x= scaledDataFrame[['buying', 'maint', 'doors','persons','lug_boot','safety']]
x.head()


# In[188]:


x.head()


# In[189]:


y.head()


# splitting data into training and testing - currently fitting data only on the training data to adjust the hyper parameters

# In[190]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train, y_test = train_test_split(x, y, test_size = 0, random_state=10)


# In[191]:


from sklearn.tree import DecisionTreeClassifier


# Playing with hyper parameters of Decision tree

# In[192]:


clf_decisionTree = DecisionTreeClassifier(criterion= "entropy", random_state=100, max_depth=5, min_samples_leaf=10)


# In[197]:


clf_decisionTree.fit(x_train, y_train)


# In[198]:


predict_decisionTree = clf_decisionTree.predict(x_train)


# In[200]:


from sklearn.metrics import classification_report, confusion_matrix
print (confusion_matrix(y_train, predict_decisionTree))
print (classification_report(y_train, predict_decisionTree))


# In[ ]:




