
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

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier


# In[2]:


import nltk

from nltk.corpus import indian
from nltk.tag import tnt

train_data = indian.tagged_sents('hindi.pos')
tnt_pos_tagger = tnt.TnT()
tnt_pos_tagger.train(train_data) #Training the tnt Part of speech tagger with hindi data


# In[3]:


path='Hindi_train/'
authors=os.listdir(path)
authors=authors[:5]
files=[]

# getting the list of files
for author in authors:
    newpath=path+author+'/'
    x=os.listdir(newpath)
    for every_file in x:
        full_path=newpath+every_file
        files=files+[full_path]


# In[4]:


#reading documents from file paths
tagged_document=[]
length=[]

for file in files:
    tagged_words=[]
    tags=[]
    doc = codecs.open(file, "r", encoding='utf-16')
    doc = doc.read()
    words=nltk.word_tokenize(doc)
    for each in words:
        tagged_word=tnt_pos_tagger.tag([each])
        tagged_words+=[tagged_word[0][1]]
    tagged_document+=[tagged_words]
    length=length+[len(doc)]
tagged_document


# In[5]:


string_representation=[]
for each in tagged_document:
    string_representation+=[" ".join(each)]


# In[6]:


vectorizer = CountVectorizer(analyzer='word')
X=vectorizer.fit_transform(string_representation)
train_data_X=pd.DataFrame(data=X.toarray(),columns=vectorizer.get_feature_names())


# In[7]:


pos=train_data_X


# In[8]:


len(files)


# In[9]:


train_data_X=[]
start=chr(0x900)                    
end=chr(0x97F)                      

vectorizer=CountVectorizer(token_pattern="["+start+"-"+end+"]+",min_df = 0)

   

  

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



# In[10]:


len(document)


# In[11]:


sum=[]
columns=[]
for each in train_data_X.columns:
    sum_of_values=train_data_X[each].sum()
    sum=sum+[sum_of_values]
    columns=columns+[each]

sum=pd.Series(data=sum,index=columns)

cut_off=train_data_X.shape[0]

function_words=sum[sum>(cut_off/2)]


X=vectorizer.fit_transform(document)
train_data_X=pd.DataFrame(data=X.toarray(),columns=vectorizer.get_feature_names())
train_data_X=train_data_X[function_words.index]

#adding additional features: no of english words per document
no_of_english_words=pd.DataFrame(no_of_english_words,columns=['no_of_english_words'])
no_of_english_words=no_of_english_words.assign(avg_word_length=avg_word_length)
no_of_english_words=no_of_english_words.assign(no_of_unique_words=no_of_unique_words)
no_of_english_words=no_of_english_words.assign(hapax=hapax)
freq_length_dist=pd.DataFrame(freq_length_dist)
no_of_english_words=pd.concat([no_of_english_words,freq_length_dist],axis=1)
no_of_english_words=no_of_english_words.div(length,axis=0)
train_data_X=pd.concat([train_data_X,no_of_english_words],axis=1)





pd.concat([pos,train_data_X],axis=1)



                          









# In[12]:


len(train_data_X)


# In[13]:


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


# In[14]:


Y=[]
for author in authors:
    newpath=path+author+'/'
    x=os.listdir(newpath)
    for every_file in x:
        Y=Y+[author]


# In[15]:


len(train_data_X)


# In[16]:


le=preprocessing.LabelEncoder()
train_data_Y=le.fit_transform(Y)


# In[17]:


model=svm.SVC()


# In[18]:


train_data_X=(train_data_X-train_data_X.min())/(train_data_X.max()-train_data_X.min())    #normalisation for learning algo
columns=train_data_X.columns[train_data_X.isnull().any()]
train_data_X=train_data_X.drop(columns,axis=1)


# In[19]:


parameters = [{'kernel': ['rbf'],
                   'gamma': [1e-4, 1e-3, 0.01, 0.1, 0.2, 0.5],
                    'C': [1, 10, 100, 1000]},
                  {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]


# In[20]:


from sklearn.naive_bayes import  MultinomialNB


# In[21]:


clf=GridSearchCV(model,param_grid=parameters,cv=10)


# In[22]:


clf=MultinomialNB()


# In[23]:


clf=cross_val_score(clf,train_data_X,train_data_Y,cv=10)


# In[24]:


clf.mean()

