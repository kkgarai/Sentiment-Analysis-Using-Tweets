from tkinter import *
import tkinter as t
from PIL import Image,ImageTk
from tkinter import ttk
import threading
import tweepy
import io
import os
from functools import partial
import csv

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from nltk.tokenize import TweetTokenizer


import re
from nltk.corpus import stopwords
import nltk
from nltk.corpus import stopwords
from  nltk.stem import SnowballStemmer
import pickle
import time


	

def resultViewer():
		
		stop_words = stopwords.words("english")
		stemmer = SnowballStemmer("english")

		tokenizer=''
		T=100
		# loading tokenizer object
		with open('tokenizer.pickle', 'rb') as handle:
			tokenizer = pickle.load(handle)

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



		model=load_model('twitter.h5')
		#print(model.summary())

		file=pd.read_csv('file.csv',header=None)


		file[0] = file[0].apply(lambda x: preprocess(x))


		t=tokenizer.texts_to_sequences(file[0])

		t=pad_sequences(t,padding='post',maxlen=T,truncating='post')


		#pred=model.predict_classes(t)
		#pred=(model.predict(t) > 0.5).astype("int32")
		pred=model.predict(t)
		list(pred.reshape(-1))

		pos=0
		neg=0
		for p in pred:
			if p>=0.5:
				pos+=1
			else:
				neg+=1
		#print(pos,neg)

		fig, axs = plt.subplots(2, 1)
		fig.suptitle('Tweet Analysis')
		axs[0].pie([pos,neg], labels = ["Positive","Negative"],colors=('green','red'),autopct='%1.1f%%')
		axs[1].plot(pred[::-1],color='blue') 
		axs[1].plot([0.5]*len(pred),'--',linewidth=0.5,color='red')

		#fig = plt.figure(figsize =(10, 5)) 
		#plt.pie([pos,neg], labels = ["Positive","Negative"],colors=('green','red'))
		#plt.plot(pred)
		plt.xlabel('Timeline')
		plt.ylabel('Sentiment')
		  
		# show plot 
		plt.show() 
		



# def loadingScreen():
# 	root = Tk()
# 	root.geometry('400x250')

# 	myProgress = ttk.Progressbar(root ,orient =  HORIZONTAL, length = 200 , mode = 'determinate' )
# 	myProgress.pack(pady = 50)

# 	myButton = Button(root , text = ' Button ' , command = getTweets).pack()

# 	root.mainloop()


class Home:
	
	def __init__(self):
		self.win = Tk()
		self.win.geometry('450x300')
		self.win.title('Sentiment Analysis')
		self.win.config(bg='White')
		self.u_frame=Frame(self.win,bg='light skyblue')
		self.u_frame.grid(row=0,column=0)
		self.icon=ImageTk.PhotoImage(Image.open('1.png'))
		self.lbl=Label(self.u_frame,text='  Sentimental Analysis  ',font='candara 30 bold',compound='left',image=self.icon,bg='light skyblue')
		self.lbl.grid(row=0,column=0)

		self.l_frame=Frame(self.win,bg='white')
		self.l_frame.grid(row=1,column=0,pady=20)

		#consumerKey,consumerSecret,accessToken,accessTokenSecret,userHandle = StringVar(),StringVar(),StringVar(),StringVar(),StringVar()
		
		#label1 = Label(self.win , font = 'candara 14', text ='Consumer Key: ').grid(row = 1 , column = 0)
		#label2 = Label(self.win , font = 'candara 14', text ='Consumer Secret: ').grid(row = 2, column = 0)
		#label3 = Label(self.win , font = 'candara 14', text ='Access Token: ').grid(row = 3, column = 0)
		#label4 = Label(self.win , font = 'candara 14', text ='Access Token Secret: ').grid(row = 4, column = 0)
		self.label4 = Label(self.l_frame , font = 'candara 14',text ='User Handle : ',bg='white').grid(row = 1, column = 0,padx=10)
		
		#Consumer_Key = Entry(self.win , textvariable = consumerKey, font = 'candara 14' ).grid(row = 1 , column = 1)
		#Consumer_Secret = Entry(self.win , textvariable = consumerSecret, font = 'candara 14').grid(row = 2 , column = 1)
		#Access_Token = Entry(self.win , textvariable = accessToken, font = 'candara 14').grid(row = 3 , column = 1)
		#Access_Token_Secret = Entry(self.win , textvariable = accessTokenSecret, font = 'candara 14').grid(row = 4 , column = 1)
		self.User_Handle = Entry(self.l_frame ,font = 'candara 12')
		self.User_Handle.grid(row = 1 , column = 1,ipady=3) 


		#Button(self.win , text = 'Get Tweets' , command = lambda :getTweets(consumerKey.get(),consumerSecret.get(),accessToken.get(),accessTokenSecret.get(),userHandle.get() ), font = 'candara 14').grid(row = 6, column = 1)
		self.bt1=Button(self.l_frame , text = 'Get Tweets' , command = self.getTweets, font = 'candara 12',bg='white').grid(row = 2, column = 1,pady=10,sticky=W)	
		# self.btn2=Button(self.win , text = 'Results' , command = resultViewer, font = 'candara 14').grid(row = 2, column = 2)
		# self.btn3=Button(self.win , text = 'R' , command = self.r, font = 'candara 14').grid(row = 3, column = 2)
		self.win = mainloop()
	# def r(self):
	# 	re()
		
		
		

		
		
	# 	self.status=True
	# 	if self.status==True:
	# 		threading.Thread(target=self.process).start()
			


#def getTweets(Consumer_Key, Consumer_Secret, Access_Token, Access_Token_Secret, User_Handle):
	def getTweets(self):
		self.myProgress = ttk.Progressbar(self.l_frame ,orient = HORIZONTAL, length = 200 , mode = 'indeterminate' )
		
		self.myProgress.grid(row=5,column=0,columnspan=2)
		self.myProgress.start(10)
		
		self.p=threading.Thread(target=self.pro).start()
		if self.p==True:
			self.myProgress.stop()

	def pro(self):	
			
			file = io.open('file.csv' , 'w' , encoding="utf-8")
			thewriter = csv.writer(file)

			#auth = tweepy.OAuthHandler(AuthInfo['1'], AuthInfo['2'])
			#auth.set_access_token(AuthInfo['3'], AuthInfo['4'])
			auth = tweepy.OAuthHandler('NRiMpFEGuGBZdYuUGqFx6jHKw', 'DyC81HQp0clndydDXwMQNXFQUEyZqxp8ISlPueBVvZiMPFtYZh')
			auth.set_access_token('842279246999506945-cZMLYn5wfddox2nE8TRMvBFbAwNhMX9', '5I3l7DhAG1gnYa09Ub3NMki9VDHZP22XKp8LxEQWxTvcs')
			
			self.api = tweepy.API(auth)
			self.value=self.User_Handle.get()
			public_tweets = self.api.user_timeline(self.value, count = 200 , include_rts = False , tweet_mode = 'extended')
			for tweet in public_tweets:
			#	print(tweet.full_text)
				thewriter.writerow([tweet.full_text])
			file.close()
			
			self.btn22=Button(self.l_frame , text = 'View Results' , command = resultViewer, font = 'candara 12',bg='white').grid(row = 6, column = 1,sticky=W,pady=10)
			self.myProgress.stop()
			return True

			
			#os.startfile('file.csv')
			
		
		#AuthInfo = {'1' : 'NRiMpFEGuGBZdYuUGqFx6jHKw' , '2' : 'DyC81HQp0clndydDXwMQNXFQUEyZqxp8ISlPueBVvZiMPFtYZh' ,
		#'3' : '842279246999506945-cZMLYn5wfddox2nE8TRMvBFbAwNhMX9' , '4' : '5I3l7DhAG1gnYa09Ub3NMki9VDHZP22XKp8LxEQWxTvcs'}
# class re:
# 	def __init__(self):
# 		self.r=Tk()
# 		self.r.geometry('300x300')
# 		self.myProgress = ttk.Progressbar(self.r,orient = HORIZONTAL, length = 200 , mode = 'indeterminate' )
# 		self.myProgress.grid(row=1,column=2)
# 		self.myProgress.start(10)	


if __name__ == "__main__":
    Home()
