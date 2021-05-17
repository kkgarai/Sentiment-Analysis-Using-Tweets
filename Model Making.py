import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input,Dense,GlobalMaxPooling1D,LSTM,Embedding,Conv1D,MaxPooling1D
from tensorflow.keras.models import Model,Sequential
from nltk.tokenize import TweetTokenizer

! wget http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip

!unzip  /content/trainingandtestdata.zip

import pandas as pd


data=pd.read_csv('training.1600000.processed.noemoticon.csv',encoding="ISO-8859-1",header=None)
data.head()

df=data[[0,5]]
df.columns=['Sentiment','Tweet']


import re
from nltk.corpus import stopwords
import nltk
from nltk.corpus import stopwords
from  nltk.stem import SnowballStemmer
nltk.download('stopwords')
stop_words = stopwords.words("english")
stemmer = SnowballStemmer("english")

TEXT_CLEANING_RE = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"


def preprocess(text,stem=False):
    text= re.sub(TEXT_CLEANING_RE,' ',str(text).lower()).strip()
    tokens = []
    for token in text.split():
        if stem:
            tokens.append(stemmer.stem(token))
        else:
            tokens.append(token)
    return " ".join(tokens)

df.Tweet = df.Tweet.apply(lambda x: preprocess(x))


df.Sentiment=df.Sentiment.apply(lambda x:1 if x==4 else 0)

x=df['Tweet']
y=np.array(df['Sentiment'],dtype=np.float).reshape(-1,1)

tokenizer=Tokenizer(num_words=20000)
tokenizer.fit_on_texts(x)
X=tokenizer.texts_to_sequences(x)
V=len(tokenizer.word_index)

X=pad_sequences(X,padding='post',maxlen=100,truncating='post')
T=X.shape[1]
print("T : ",T,'V : ',V)



X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=69)

# Building the model
model=Sequential([
                Input(shape=(T,)),
                Embedding(V+1,20),

                Conv1D(filters=32,kernel_size=3,activation='relu'),
                MaxPooling1D(pool_size=3),

                Conv1D(filters=64,kernel_size=3,activation='relu'),
                MaxPooling1D(pool_size=3),

                Conv1D(filters=128,kernel_size=3,activation='relu'),
                MaxPooling1D(pool_size=3),

                GlobalMaxPooling1D(),
                Dense(units=1,activation='sigmoid')
])

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

r=model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=1)




plt.plot(r.history['accuracy'],label='accuracy')
plt.plot(r.history['val_accuracy'],label='val_accuracy')
plt.legend()

plt.plot(r.history['loss'],label='loss')
plt.plot(r.history['val_loss'],label='val_loss')
plt.legend()