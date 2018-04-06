
# coding: utf-8

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import os
from sklearn import svm, model_selection, tree, preprocessing, metrics
from sklearn.ensemble import RandomForestClassifier
import codecs
from nltk.probability import FreqDist
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV
import helper


path='/mnt/A042994142991CDA/Hindi_train/'
svm_scores=[]
random_forest_scores=[]
svm_times=[]
random_forest_times=[]


vectorizer = CountVectorizer(analyzer='char_wb', ngram_range=(2,3), min_df = 0)

for i in range(3,15):

    train_data_X,train_data_Y,length=helper.prepare_data(vectorizer=vectorizer,add_features='no',path=path,no_of_authors=i)


    helper.text_normalise(train_data_X,length)  #tf-idf normalisation
    helper.feature_normalise(train_data_X)      #normalisation for machine learning algo

    svm_score,svm_time,random_forest_score,random_forest_time=helper.learn(train_data_X,train_data_Y)
    svm_scores+=[svm_score]
    svm_times+=[svm_time]
    random_forest_scores+=[random_forest_score]
    random_forest_times+=[random_forest_time]
    















# In[57]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[58]:


plt.plot([x for x in range (3,15)],svm_scores,[x for x in range (3,15)],random_forest_scores)


# In[59]:


plt.plot([x for x in range (3,15)],svm_times,[x for x in range (3,15)],random_forest_times)

