# Read data from file and get all agent information

import gensim, os, re, csv, codecs, sys, nltk

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

from string import punctuation

from gensim.models import KeyedVectors

from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers.merge import concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.regularizers import l2,l1,l1_l2

#print("On a fresh system it takes around 4 min download gensim / nltk libraries")

# Get stopwords from nltk library
nltk.download('stopwords')

# Get agent conversations and filter duplicates
def read_data():
    df = pd.read_csv('data/data.csv',sep='|', quotechar='"')
    df_agent = df.loc[(df.is_customer == 'f')]
    df_agent.skill_name.fillna(value='none',inplace=True)
    df_agent_unique = df_agent.drop_duplicates('content')
    df_agent_unique=df_agent_unique.filter(items=['content','skill_name','id'])
    return df_agent_unique
    
# Remove very short sentences
def clean_up(df):
    df = df[~(df.content.str.len() < 20)&(df['content'].notnull())]
    return df

# Balance data by limiting the min count
def drop_tail(df, min_count=2000):
    vc = df['skill_name'].value_counts()
    keep_skills=set(vc[vc > min_count].index)
    df=df[df['skill_name'].isin(keep_skills)]
    return df

# Balance data by limiting the max count
def drop_head(df, max_count=15000):
    df = df.sample(frac=1).groupby('skill_name', sort=False).head(max_count)
    return df

# Create dummies for dataframe
def get_dummies(df):
    df = pd.get_dummies(pd.DataFrame(df))
    df = df.reset_index(drop=True)
    return df

# Split data to train / test 
def create_train_test(df, test_size_value=0.2):
    X = df['content']
    y = df['skill_name']
    X_train, X_test, y_train, y_test = train_test_split(X,y,
                                                    test_size=test_size_value,random_state=42,stratify=y)
    return X_train, X_test, y_train, y_test
    

# Data Processing techniques (option to remove stop words / stem words) 
def text_to_wordlist(text, remove_stopwords=True, stem_words=False):
    
    # Convert words to lower case and split them
    text = text.lower().split()

    # Optionally, remove stop words
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]
    
    text = " ".join(text)

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    
    # Optionally, shorten words to their stems
    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)
    
    # Return a list of words
    return(text)


print("Done importing helper_functions")

