
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
svm_scores=[]
random_forest_scores=[]
svm_times=[]
random_forest_times=[]


start=chr(0x900)                    # 0x900 is the unicode point of first character of hindi alphabet
end=chr(0x97F)                      # 0x900 is the unicode point of last character of hindi alphabet


vectorizer=CountVectorizer(token_pattern="["+start+"-"+end+"]+",min_df = 0,binary=True)


authors=os.listdir(path)
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
for file in files:
        doc = codecs.open(file, "r", encoding='utf-16')
        doc = doc.read()
        document=document+[doc]
        
#transforming data into feature vector
X=vectorizer.fit_transform(document)
train_data_X=pd.DataFrame(data=X.toarray(),columns=vectorizer.get_feature_names())


#extracting words which have occured in  more than half of the documents i,e most frequent words
#The extracted words have a very high probability of being function words

sum=[]
columns=[]
for each in train_data_X.columns:
        sum_of_values=train_data_X[each].sum()
        sum=sum+[sum_of_values]
        columns=columns+[each]

sum=pd.Series(data=sum,index=columns)
cut_off=train_data_X.shape[0]
function_words=sum[sum>(cut_off/2)]    #cut off frequency i,e percentage of total documents the word appeared. Here we set to 50%.



for i in range(5,6):
    
    try:
        os.mkdir('function_words')
    except FileExistsError:
         pass
        
        
    
    file=open('function_words/results for '+str(i)+' authors','w')
    iterations=0
    x=int(sp.comb(14,i))
    if(x>20):
        iterations=20
    else:
        iterations=x
    iterations=20
 

    for a in range(iterations):
        vect=CountVectorizer(token_pattern="["+start+"-"+end+"]+",min_df = 0)
        random_list=random.sample(range(14),i)
        train_data_X,train_data_Y,length=helper.prepare_data(authors_to_consider=random_list,vectorizer=vect,add_features='no',path=path,no_of_authors=i)
        
    
        train_data_X=train_data_X[function_words.index]

    
        helper.text_normalise(train_data_X,length)  #tf-idf normalisation
        helper.learn(train_data_X,train_data_Y,file)

    file.close()
    

 


   


    


