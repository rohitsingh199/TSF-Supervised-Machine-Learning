#!/usr/bin/env python
# coding: utf-8

# # Task 4 - To Explore Decision Tree Algorithm.For the given ‘Iris’ dataset

# import libraries

# 
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Loading dataset

# In[2]:


from sklearn.datasets import load_iris
dat = load_iris()
data = pd.DataFrame(dat.data,columns = dat.feature_names)
y = dat.target


# In[3]:



# Explore the data
data.head()


# In[4]:


# Shape of target data
y.shape


# In[5]:



# Check the column names
data.columns


# In[6]:


#Check null values
data.isnull().sum()


# # Visualisation

# In[7]:


data.hist(figsize = (15,15))


# In[8]:


sns.pairplot(data)


# # Build the model

# In[9]:


x = data
y = y
from sklearn .model_selection import train_test_split 
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.30,random_state = 42)

print("Shape of feature training data :",x_train.shape)
print("Shape of target training data :",y_train.shape)
print("Shape of feature test data :",x_test.shape)
print("Shape of target test data :",y_test.shape)


# In[10]:


from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import accuracy_score,confusion_matrix

clf = DecisionTreeClassifier()
clf.fit(x_train,y_train)


# In[11]:


# Accuracy For training data
predict = clf.predict(x_train)
print("Accuracy of training data : ",accuracy_score(predict,y_train)*100,"%")
print("Confusin matrix of training data :'\n' ",confusion_matrix(predict,y_train))
sns.heatmap(confusion_matrix(predict,y_train),annot = True,cmap = 'BuGn')


# In[12]:



# Accuracy For testing data
predict = clf.predict(x_test)
print("Accuracy of testing data : ",accuracy_score(predict,y_test)*100,"%")
print("Confusin matrix of testing data :\n ",confusion_matrix(predict,y_test))
sns.heatmap(confusion_matrix(predict,y_test),annot = True,cmap = 'BuGn')


# In[13]:


plt.figure(figsize  = (18,18))
tree.plot_tree(clf,filled = True,rounded = True,proportion = True,node_ids = True , feature_names = dat.feature_names)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




