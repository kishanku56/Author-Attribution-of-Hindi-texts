
# coding: utf-8

# In[1]:


from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import os
from sklearn import svm, model_selection, tree, preprocessing, metrics
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import codecs
from nltk.probability import FreqDist
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV
import time
import helper
import scipy.special as sp
import random

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import shutil


# In[2]:


path='Hindi_train/'
svm_scores=[]
random_forest_scores=[]
svm_times=[]
random_forest_times=[]


start=chr(0x900)                    # 0x900 is the unicode point of first character of hindi alphabet
end=chr(0x97F)                      # 0x900 is the unicode point of last character of hindi alphabet


vectorizer=CountVectorizer(token_pattern="["+start+"-"+end+"]+",min_df = 0)


for i in range(5,6):
    try:
        os.mkdir('bag of words')
    except FileExistsError:
         pass
        
        
    
    file=open('bag of words/results for '+str(i)+' authors','w')
    iterations=0
    x=int(sp.comb(14,i))
    if(x>20):
        iterations=20
    else:
        iterations=x
        
    
    x1=0
    x2=0
    y1=0
    y2=0
    for a in range(iterations):
        random_list=random.sample(range(14),5)
        train_data_X,train_data_Y,length=helper.prepare_data(authors_to_consider=random_list,vectorizer=vectorizer,add_features='no',path=path,no_of_authors=i)

        helper.text_normalise(train_data_X,length)  #tf-idf normalisation
        helper.learn(train_data_X,train_data_Y,file)

    
    
   # helper.mean_accuracy(iterations,file)
    file.close()


