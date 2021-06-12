#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


zoo = pd.read_csv("C:/Users/vinay/Downloads/Zoo.csv")
zoo


# In[3]:


zoo.rename(columns = {"animal name":"animalname"},inplace =True)


# In[4]:


zoo


# In[5]:


ani =zoo["animalname"].value_counts()
ani


# ## animal name is the column which is categorical in nature. So, we converted into dummy variables
# ## if we look at all the columns, all the variables except legs are binary values in nature. 
# ## Legs column is actually a set of values which being repeated. So, we tend to factorize the variable.
# 

# In[6]:


zoo["type"].value_counts()
data = zoo.describe()
zoo.info()


# In[7]:


##and even the standard deviation is 2.03
zoo["legs"].var() ### 4.13


# In[8]:


zoo["legs"].var() ### 4.13
zoo["hair"].var()## 0.24
zoo["feathers"].var()## 0.16


# In[9]:


zoo["eggs"].var()  ##0.2453
zoo["airborne"].var() ###0.183
zoo["milk"].var()  ###0.243


# ## Creating variance dataframe

# In[10]:


variance = zoo.var()
variance


# ## The minimum of all the vaiables except legs is'0' and the maximum value of the variables except legs is 1.
# ## Legs minimum value is zero and maximum value is 8

# ## Animal name has discreate, categorical, non numeric data. So, we convert them into dummy variables.

# In[11]:


##Creating dummy variables for the animal variable 
dummy = pd.get_dummies(zoo["animalname"],drop_first =True)
dummy


# In[12]:


zoo = pd.concat([zoo,dummy],axis=1)


# In[13]:


zoo = zoo.drop(["animalname"],axis=1)
zoo


# In[18]:


zoo["legs"],_=pd.factorize(zoo["legs"])
zoo["legs"],_


# In[21]:


labels = zoo.iloc[:,16]
labels


# In[22]:


features = zoo.drop(["type"],axis=1)
features


# ## Normalizing the equation

# In[23]:


def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return(x)


# In[25]:


data = norm_func(features)
data


# In[27]:


##Splitting the data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(data,labels,test_size =0.2, stratify=labels)


# In[29]:


from sklearn.neighbors import KNeighborsClassifier as KN
model1 =KN(n_neighbors = 5)
model1.fit(x_train,y_train)


# In[30]:


##Accuracy on the training data
train_acc = np.mean((model1.predict(x_train)==y_train))
train_acc   ###95.00%


# In[32]:


##Accuracy on test data
test_Acc = np.mean(model1.predict(x_test)==y_test)
test_Acc


# ## Trying for k=7

# In[33]:


model2 = KN(n_neighbors=7)
model2.fit(x_train,y_train)


# In[34]:


train2_acc = np.mean(model2.predict(x_train)==y_train)
train2_acc


# In[35]:


test2_acc = np.mean(model2.predict(x_test)==y_test)
test2_acc


# In[36]:


###Creating a empty list
acc=[]


# ## running KNN algorithm for 7 to 50 nearest neighbours and 
# ## storing the accuracy values 
# 

# In[37]:


for i in range(7,50,2):
    model2=KN(n_neighbors = i)
    model2.fit(x_train,y_train)
    train_acc = np.mean(model2.predict(x_train)==y_train)
    test_acc = np.mean(model2.predict(x_test)==y_test)
    acc.append([train_acc,test_acc])


# In[39]:


import matplotlib.pyplot as plt
plt.plot(np.arange(7,50,2),[i[0] for i in acc],"bo-")


# In[40]:


##test accuracy plot
plt.plot(np.arange(7,50,2),[i[1] for i in acc],"ro-")    
plt.legend(["train_acc", "test_acc"])


# ## The plot shows k=17,Trying for k=17

# In[42]:


model_fin = KN(n_neighbors = 17) 
model_fin.fit(x_train,y_train)


# In[47]:


train_fin = np.mean(model_fin.predict(x_train)==y_train)
train_fin


# In[45]:


test_fin =np.mean(model_fin.predict(x_test)==y_test)
test_fin


# In[ ]:




