
# coding: utf-8

# In[75]:


import pandas as pd


# In[94]:


def mean_accuracy(iterations,file):
    data=pd.read_csv('bag of words/results for 5 authors')

    for x in range(5):
        index=[5*i+x for i in range (iterations)]
        temp=data.iloc[index]
        print(temp.iloc[0][0])
        print(temp.mean())


# In[95]:


mean_accuracy(20,x)

