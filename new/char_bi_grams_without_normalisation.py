
# coding: utf-8

# In[1]:


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
import scipy.special as sp
import random


path='Hindi_train/'


bigram_vectorizer=CountVectorizer(analyzer='char_wb', ngram_range=(2,2), min_df= 0)
trigram_vectorizer = CountVectorizer(analyzer='char_wb', ngram_range=(3,3), min_df= 0)
tetra_gram_vectorizer= CountVectorizer(analyzer='char_wb', ngram_range=(4,4), min_df= 0)
bi_tri_gram_vectorizer=CountVectorizer(analyzer='char_wb', ngram_range=(2,3), min_df= 0)
bi_tri_tetra_gram_vectorizer=CountVectorizer(analyzer='char_wb', ngram_range=(2,4), min_df= 0)



vectorizers=[bigram_vectorizer,trigram_vectorizer,tetra_gram_vectorizer,bi_tri_gram_vectorizer,bi_tri_tetra_gram_vectorizer]



for vectorizer in vectorizers:
    
    
    svm_scores=[]
    random_forest_scores=[]
    svm_times=[]
    random_forest_times=[]

    for i in range(5,6):
        iterations=0
        x=int(sp.comb(14,i))
        if(x>20):
            iterations=20
        else:
            iterations=x
        iterations=1

        x1=0
        x2=0
        y1=0
        y2=0
        for a in range(iterations):
            random_list=random.sample(range(5),i)
            train_data_X,train_data_Y,length=helper.prepare_data(authors_to_consider=random_list,vectorizer=vectorizer,add_features='no',path=path,no_of_authors=i)


            
            helper.feature_normalise(train_data_X)      #normalisation for machine learning algo

            a,b,c,d=helper.learn(train_data_X,train_data_Y)
            x1+=a
            x2+=b
            y1+=c
            y2+=d

        x1=x1/iterations
        x2=x2/iterations
        y1=y1/iterations
        y2=y2/iterations

        svm_scores+=[x1]
        svm_times+=[x2]
        random_forest_scores+=[y1]
        random_forest_times+=[y2]
        print(vectorizer)
        print("random_forest",random_forest_scores)
        print("svm",svm_scores)














