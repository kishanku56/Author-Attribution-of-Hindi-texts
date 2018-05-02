
# coding: utf-8

# In[20]:


import pandas as pd
bow = pd.read_csv("bag of words/results for 5 authors")

for x in range(5):
    index=[5*i+x for i in range(20)]
    temp=bow.iloc[index]
    print(temp.iloc[index[0]][0])
    print(temp.mean())
    print("\n")
    


# In[21]:


bow = pd.read_csv("bigrams/results for 5 authors")

for x in range(5):
    index=[5*i+x for i in range(20)]
    temp=bow.iloc[index]
    print(temp.iloc[index[0]][0])
    print(temp.mean())
    print("\n")


# In[23]:


bow = pd.read_csv("trigrams/results for 5 authors")

for x in range(5):
    index=[5*i+x for i in range(20)]
    temp=bow.iloc[index]
    print(temp.iloc[index[0]][0])
    print(temp.mean())
    print("\n")


# In[26]:


bow = pd.read_csv("bi_trigrams/results for 5 authors")

for x in range(5):
    index=[5*i+x for i in range(20)]
    temp=bow.iloc[index]
    print(temp.iloc[index[0]][0])
    print(temp.mean())
    print("\n")


# In[27]:


bow = pd.read_csv("function_words/results for 5 authors")

for x in range(5):
    index=[5*i+x for i in range(20)]
    temp=bow.iloc[index]
    print(temp.iloc[index[0]][0])
    print(temp.mean())
    print("\n")


# In[28]:


bow = pd.read_csv("function_words and trigrams/results for 5 authors")

for x in range(5):
    index=[5*i+x for i in range(20)]
    temp=bow.iloc[index]
    print(temp.iloc[index[0]][0])
    print(temp.mean())
    print("\n")

