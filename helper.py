
# coding: utf-8

# In[1]:


from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import os
from sklearn import svm, model_selection, tree, preprocessing, metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import codecs
from nltk.probability import FreqDist
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV
import time

    
le=preprocessing.LabelEncoder() 


def prepare_data(authors_to_consider,vectorizer,path,add_features='yes',drop_columns='yes',no_of_authors=14):
    
    authors=os.listdir(path);
    temp=[]
    for i in range(len(authors_to_consider)):
        temp+=[authors[authors_to_consider[i]]]
    
 
    authors=temp
    
    files=[]

    #getting the list of files
    for author in authors:
        newpath=path+author+'/'
        x=os.listdir(newpath)
        for every_file in x:
            full_path=newpath+every_file
            files=files+[full_path]

    #reading documents from file paths
    document=[]
    length=[]
    hapax=[]
    word_count=[]
    no_of_english_words=[]                                                  # number of english words that an author uses
    avg_word_length=[]                                                      # length of document divided by number of words
    no_of_unique_words=[]                                                   # vocabulary richness
    freq_length_dist=[]
    for file in files:
        doc = codecs.open(file, "r", encoding='utf-16')
        doc = doc.read()
        words=doc.split(' ')
        freq_dist=FreqDist(words)
        no_of_hapax=len(freq_dist.hapaxes())
        hapax=hapax+[no_of_hapax]
        freq_of_different_words=[0 for i in range(15)]
        unique_words=set(words)
        no_of_unique_words=no_of_unique_words+[len(unique_words)]
        no_of_words=len(words)
        word_count = word_count + [no_of_words]
        avg_word_length=avg_word_length+[len(doc)/no_of_words]
        english_words=[]
        for each_word in words:
            if(len(each_word)>14):
                words.remove(each_word)
                continue;
            freq_of_different_words[len(each_word)]=freq_of_different_words[len(each_word)]+1
            if((len(each_word)>0) and (ord(each_word[0])<=126)):
                english_words=english_words+[each_word]
        freq_length_dist+=[freq_of_different_words]
        no_of_english_words=no_of_english_words+[len(english_words)]
        length=length+[len(doc)]
        document=document+[doc]

    #transforming data into feature vector
    X=vectorizer.fit_transform(document)
    train_data_X=pd.DataFrame(data=X.toarray(),columns=vectorizer.get_feature_names())
    
    
    #dropping columns which have numeric parts
    if(drop_columns=='yes'):
        dropped_columns=[]
        for each in train_data_X.columns:
            if (each.isalpha()==False):
                dropped_columns=dropped_columns+[each]

        train_data_X=train_data_X.drop(dropped_columns,axis=1)

    
    Y=[]
    for author in authors:
        newpath=path+author+'/'
        x=os.listdir(newpath)
        for every_file in x:
            Y=Y+[author]

    train_data_Y=le.fit_transform(Y)
    
    if(add_features=='no'):
        return (train_data_X,train_data_Y,length)
    
    #adding additional features: no of english words per document
    no_of_english_words=pd.DataFrame(no_of_english_words,columns=['no_of_english_words'])
    no_of_english_words=no_of_english_words.assign(avg_word_length=avg_word_length)
    no_of_english_words=no_of_english_words.assign(no_of_unique_words=no_of_unique_words)
    no_of_english_words=no_of_english_words.assign(hapax=hapax)
    freq_length_dist=pd.DataFrame(freq_length_dist)
    no_of_english_words=pd.concat([no_of_english_words,freq_length_dist],axis=1)
    no_of_english_words=no_of_english_words.div(length,axis=0)
    train_data_X=pd.concat([train_data_X,no_of_english_words],axis=1)





    train_data_X=train_data_X.assign(l=length)
    train_data_X=train_data_X.assign(word_count=word_count)
    
    return (train_data_X,train_data_Y,length)
    
    
def text_normalise(train_data_X,length):
    x,y=train_data_X.shape
    #term frequency normalization
    train_data_X=train_data_X.div(length,axis=0)

    #calculating idf for each column
    import math as math
    l=[]
    for each in train_data_X.columns:
        document=0
        for value in train_data_X.loc[:,each]:
            if(value!=0):
                document=document+1
        data=math.log(x/document)
        l=l+[data]

    #tf idf
    train_data_X=train_data_X.mul(l,axis=1)
    
def feature_normalise(train_data_X):
    train_data_X=(train_data_X-train_data_X.mean())/(train_data_X.max()-train_data_X.min())    #normalisation for learning algo
    columns=train_data_X.columns[train_data_X.isnull().any()]
    train_data_X=train_data_X.drop(columns,axis=1)
    
    
    
def learn(train_data_X,train_data_Y):
    model=[svm.SVC(),RandomForestClassifier()]
    

    svm_parameters = [{'kernel': ['rbf'],
                   'gamma': [1e-4, 1e-3, 0.01, 0.1, 0.2, 0.5],
                    'C': [1, 10, 100, 1000]},
                  {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

    Forest_parameters=[{'n_estimators':[1000],'max_depth':[5,6,7,8,9,10]}]

    parameters=[svm_parameters,Forest_parameters]

    best_scores=[]


    k=0
    for each_model in model:
        parameter=parameters[k]

        clf=GridSearchCV(each_model,param_grid=parameter,n_jobs=2,cv=10)
        start_time=time.time()
        clf.fit(train_data_X,train_data_Y)
        end_time=time.time()
        if(k==0):
            svm_score=clf.best_score_
            svm_time=end_time-start_time

        else:
            random_forest_score=clf.best_score_
            random_forest_time=end_time-start_time
        k+=1

    return svm_score,svm_time,random_forest_score,random_forest_time

