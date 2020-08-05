import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from statistics import mean 
from collections import Counter

# NLP
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer

# Sentiment Analysis for english
from textblob import TextBlob
# Sentiment Analysis For French
from textblob_fr import PatternTagger, PatternAnalyzer


class whatsapp_analyzer():
	""" main class to declare """
	def __init__(self,text_file,language = 'french',top_x_common_words = 30):
		self.file = text_file
		self.language = language
		self.df_drop_duplicates_day = None
		self.first_user = None
		self.second_user = None
		self.top_x_words = top_x_common_words

	def read_file(self):
		"""read raw file
		input .txt file
		output pandas dataframe"""

		file1 = open(self.file,"r")
		df = pd.DataFrame(data = list(file1.read().split("[")), columns = ['raw_message'])
		file1.close()
		return df 

	def apply_preprocessing(self,df,specific_preprocessing):

		"""create various column useful for analysis
		input raw pandas dataframe
		output pandas dataframe with about 10 new columns"""

		df['raw_date'] = df['raw_message'].apply(lambda row: row.split("]")[0])
		df['datetime'] = df['raw_date'].apply(lambda row: dt.datetime.strptime(row, '%m/%d/%y, %I:%M:%S %p'))
		df['actual_date'] = df['datetime'].apply(lambda row: row.date())
		df['hour'] = df['datetime'].apply(lambda row: row.hour)
		df['weekday'] = df['datetime'].apply(lambda row: row.weekday()) # Watch out, Monday is 0, Tuesday is 1 etc
		df['actual_message'] = df['raw_message'].apply(lambda row: " ".join(row.split("]")[1:])[0:])
		df['sender'] = df['actual_message'].apply(lambda row: row.split(':')[0])
		df['actual_message'] = df['actual_message'].apply(lambda row: " ".join(row.split(':')[1:])[0:])
		df['actual_message']  = df['actual_message'].apply(lambda row: row.replace('\n',''))
		df['message_lenght'] = df['actual_message'].apply(lambda row: len(row.split(' ')))
		df['weekend'] = 0
		df.loc[df.weekday > 4, 'weekend'] = 1

		# Isolating the name of the unique users
		self.first_user = df['sender'].unique()[0]
		self.second_user = df['sender'].unique()[1]
		
		if specific_preprocessing:
			df['sender'] = df['sender'].map({self.first_user:"user_1",self.second_user:"user_2"},)
			self.first_user = "user_1"
			self.second_user = "user_2"

		# Creating a first_message column
		self.df_drop_duplicates_day = df.drop_duplicates(subset=['actual_date'],keep = 'first').copy()
		self.df_drop_duplicates_day['first_message'] = 1
		df = pd.merge(df, self.df_drop_duplicates_day['first_message'],how = 'left',right_index= True, left_index= True)
		df['first_message'].fillna(0,inplace= True)

		return df 

	def average_response_time(self,df):
		"""calculate the average time it takes to each users to answer
		input: pandas dataframe
		output: tuple"""
		user_1_response_time = []
		user_2_response_time = []

		for i in range(0,df.shape[0]-1):
		    if (df['sender'].iloc[i+1] != df['sender'].iloc[i] and df['first_message'].iloc[i+1] != 1):
		        time_delta =  df['datetime'].iloc[i+1] - df['datetime'].iloc[i]
		        time_difference = time_delta.seconds
		        if df['sender'].iloc[i+1] == self.first_user :
		            user_1_response_time.append(time_difference)
		        else:
		            user_2_response_time.append(time_difference)
		return (mean(user_1_response_time)/60, mean(user_2_response_time)/60)

	def number_of_days_without_conversation(self, df):
		"""Identify the longest period of times where no messages where exchanged
		input: pandas dataframe
		output : head of pandas dataframe"""

		all_days = [day.date() for day in pd.date_range(min(self.df_drop_duplicates_day['datetime']), max(self.df_drop_duplicates_day['datetime']), freq='D')]
		days_present = list(df['actual_date'].unique())
		days_where_no_messages_exchanged = [day for day in all_days if day not in days_present]

		max_day_without_speaking = 1
		day_without_conv = {}

		for i,day in enumerate(days_where_no_messages_exchanged):
		    try: 
		        if days_where_no_messages_exchanged[i+1] == day + dt.timedelta(days=1):
		            max_day_without_speaking += 1
		        else: 
		            # end of consecutive serie, setting the final count for the latest serie
		            day_without_conv[day] = max_day_without_speaking
		            # reseting the value of consecutive days to one
		            max_day_without_speaking = 1
		    except IndexError: 
		        pass        

		df_consecutive_days_without_talking = pd.DataFrame(index = day_without_conv.keys(),data=day_without_conv.values(),columns = ['number of consecutive days without talking']).sort_values(ascending=False, by = ['number of consecutive days without talking']).reset_index().rename({'index':'end_date'},axis = 1)

		return df_consecutive_days_without_talking.head(10)

	def number_of_deleted_messages(self,df):
		"""Simply compute the number of deleted messages
		input: pandas dataframe
		output: pandas dataframe"""

		df_deleted_message = df[df['actual_message'].str.contains('You deleted this message.')|df['actual_message'].str.contains('This message was deleted.')]
		df_number_deleted_messages = pd.DataFrame(df_deleted_message.groupby(by= ['sender'])['raw_message'].count())

		return df_number_deleted_messages

	def number_of_missed_voice_call(self,df):
		"""Simply compute the number of missed voice calls
		input: pandas dataframe
		output: pandas dataframe"""

		df_missed_voice_call = df[df['actual_message'].str.contains('Missed voice call')]
		df_missed_voice_call = pd.DataFrame(df_missed_voice_call.groupby(by= ['sender'])['raw_message'].count())

		return df_missed_voice_call

	def most_common_words_used(self, df):
		"""Compute the most common words used by each users while removing the stop words
		input : pandas dataframe
		output : pandas dataframe"""

		# Removing the messages refering to missed voice calls as these should not account in the counting of most common words
		df = df[~df['actual_message'].str.contains('Missed voice call')]

		user_1_words = "".join(list(df[df['sender'] == self.first_user]['actual_message']))
		user_2_words = "".join(list(df[df['sender'] == self.second_user]['actual_message']))

		user_1_unique_words = [w for w in word_tokenize(user_1_words.lower(),language = self.language) if w.isalpha()]
		user_2_unique_words = [w for w in word_tokenize(user_2_words.lower(),language = self.language) if w.isalpha()]

		user_1_unique_words_no_stops = [t for t in user_1_unique_words if t not in stopwords.words(self.language)]
		user_2_unique_words_no_stops = [t for t in user_2_unique_words if t not in stopwords.words(self.language)]

		user_1_counter = Counter(user_1_unique_words_no_stops)
		user_2_counter = Counter(user_2_unique_words_no_stops)

		return (user_1_counter.most_common(self.top_x_words),user_2_counter.most_common(self.top_x_words))

	def visualize(self,df):
		"""generate analysis visualization and store them in a file
		input : pandas dataframe
		output : jpeg files """

		print("Total Number of messages sent")
		print(pd.DataFrame(df.groupby(by = ['sender']).count()['raw_message']).sort_values(by = ['raw_message'], ascending=False))
		print("________________________________________________________________________________")

		plt.title("Distribution of total messages sent during the week")
		plt.plot(pd.DataFrame(df.groupby(by = ['weekday']).count()['raw_message']))
		plt.show()
		
		try: 
			# More convenient if open in Jupyter version but won't work in terminal
			display(pd.DataFrame(df.groupby(by = ['weekday']).count()['raw_message']).T)
		except: 
			# Adapted to terminal display
			print(pd.DataFrame(df.groupby(by = ['weekday']).count()['raw_message']).T)
		print("________________________________________________________________________________")

		plt.plot(pd.DataFrame(df.groupby(by = ['actual_date']).count()['raw_message']))
		plt.title("Distribution of total daily messages sent so far")
		plt.show()
		print("________________________________________________________________________________")

		plt.plot(pd.DataFrame(df[df['weekend']==0].groupby(by = ['hour']).count()['raw_message']))
		plt.title("Distribution of total hourly messages sent during WEEKDAYS")
		plt.show()
		try: 
			# More convenient if open in Jupyter version but won't work in terminal
			display(pd.DataFrame(df[df['weekend']==0].groupby(by = ['hour']).count()['raw_message']).T)
		except: 
			# Adapted to terminal display
			print(pd.DataFrame(df[df['weekend']==0].groupby(by = ['hour']).count()['raw_message']).T)
		print("________________________________________________________________________________")

		plt.plot(pd.DataFrame(df[df['weekend']==1].groupby(by = ['hour']).count()['raw_message']))
		plt.title("Distribution of total hourly messages sent during WEEKEND")
		plt.show()
		try: 
			# More convenient if open in Jupyter version but won't work in terminal
			display(pd.DataFrame(df[df['weekend']==1].groupby(by = ['hour']).count()['raw_message']).T)
		except: 
			# Adapted to terminal display
			print(pd.DataFrame(df[df['weekend']==1].groupby(by = ['hour']).count()['raw_message']).T)
		print("________________________________________________________________________________")

		print("Number of first messages of the day sent")
		try: 
			# More convenient if open in Jupyter version but won't work in terminal
			display(pd.DataFrame(self.df_drop_duplicates_day.groupby(by = ['sender'])['raw_message'].count()).sort_values(by = ['raw_message'], ascending=False))
		except: 
			# Adapted to terminal display
			print(pd.DataFrame(self.df_drop_duplicates_day.groupby(by = ['sender'])['raw_message'].count()).sort_values(by = ['raw_message'], ascending=False))
		print("________________________________________________________________________________")

		print("Average response time from each users (minutes)")
		response_time = self.average_response_time(df)
		try: 
			# More convenient if open in Jupyter version but won't work in terminal
			display(pd.DataFrame(data = [response_time[0],response_time[1]], index = [self.first_user ,self.second_user],columns = ['average response time']))
		except: 
			# Adapted to terminal display
			print(pd.DataFrame(data = [response_time[0],response_time[1]], index = [self.first_user ,self.second_user],columns = ['average response time']))
		print("________________________________________________________________________________")

		print("Number of consecutive days without talking")
		print(self.number_of_days_without_conversation(df))
		print("________________________________________________________________________________")

		print("Number of deleted messages")
		print(self.number_of_deleted_messages(df))
		print("________________________________________________________________________________")

		print("Number of missed voice calls (only available for the user who downloaded the conversation")
		print(self.number_of_missed_voice_call(df))
		print("________________________________________________________________________________")

		print(f"Most common words used by {self.first_user}")
		try: 
			# More convenient if open in Jupyter version but won't work in terminal
			display(pd.DataFrame(self.most_common_words_used(df)[0],columns= ['word','count']))
		except: 
			# Adapted to terminal display
			print(pd.DataFrame(self.most_common_words_used(df)[0],columns= ['word','count']))
		print("________________________________________________________________________________")

		print(f"Most common words used by {self.second_user}")
		try: 
			# More convenient if open in Jupyter version but won't work in terminal
			display(pd.DataFrame(self.most_common_words_used(df)[1],columns= ['word','count']))
		except: 
			# Adapted to terminal display
			print(pd.DataFrame(self.most_common_words_used(df)[1],columns= ['word','count']))
		print("________________________________________________________________________________")

		return 0

	def analyse(self,specific_preprocessing = True):
		"""main method
		input :
		specific_preprocessing (optional) - Boolean - 
		output : display of pandas dataframe and various plots 
		"""
		# read file
		df = self.read_file()
		
		if specific_preprocessing :
			# removing ']' special character
			df.drop([0,3531],inplace = True)

		# create various columns
		df = self.apply_preprocessing(df,specific_preprocessing)

		return self.visualize(df)

if __name__ == '__main__':

	analyser = whatsapp_analyzer("_chat.txt",language = 'french',top_x_common_words = 30)
	df = analyser.analyse()

	print(df)
