# Tweets gathering and sentiment analysis

## 1) Gathering stage

#Retrieving and cleaning tweets
import time
import json
import csv
import os
import pandas as pd
import tweepy
import re
import string
from textblob import TextBlob
import io
import unittest
import preprocessor as p

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# consumer keys and access tokens, used for OAuth
consumer_key="----------"
consumer_secret="----------"
access_token="----------"
access_token_secret="----------"
 
# OAuth process, using the keys and tokens
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
 
# creation of the actual interface, using authentication
api = tweepy.API(auth)

#declare file paths as follows forfiles
telemedicine_tweets = "telemedicine_data4.csv"
epilepsy_tweets = "epilepsy_data.csv"
heart_stroke_tweets = "heart_stroke_tweets_data.csv"

#set two date variables for date range
start_date = '2020-02-24'
end_date = '2018-02-25'

#columns of the csv file
COLS = ['id', 'created_at', 'source', 'original_text','clean_text', 'sentiment','polarity','subjectivity', 'lang',
'favorite_count', 'retweet_count', 'original_author',   'possibly_sensitive', 'hashtags',
'user_mentions', 'place', 'place_coord_boundaries']

#HappyEmoticons
emoticons_happy = set([
    ':-)', ':)', ';)', ':o)', ':]', ':3', ':c)', ':>', '=]', '8)', '=)', ':}',
    ':^)', ':-D', ':D', '8-D', '8D', 'x-D', 'xD', 'X-D', 'XD', '=-D', '=D',
    '=-3', '=3', ':-))', ":'-)", ":')", ':*', ':^*', '>:P', ':-P', ':P', 'X-P',
    'x-p', 'xp', 'XP', ':-p', ':p', '=p', ':-b', ':b', '>:)', '>;)', '>:-)',
    '<3'
    ])

# Sad Emoticons
emoticons_sad = set([
    ':L', ':-/', '>:/', ':S', '>:[', ':@', ':-(', ':[', ':-||', '=L', ':<',
    ':-[', ':-<', '=\\', '=/', '>:(', ':(', '>.<', ":'-(", ":'(", ':\\', ':-c',
    ':c', ':{', '>:\\', ';('
    ])


#Emoji patterns
emoji_pattern = re.compile("["
         u"\U0001F600-\U0001F64F"  # emoticons
         u"\U0001F300-\U0001F5FF"  # symbols & pictographs
         u"\U0001F680-\U0001F6FF"  # transport & map symbols
         u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
         u"\U00002702-\U000027B0"
         u"\U000024C2-\U0001F251"
         "]+", flags=re.UNICODE)

#combine sad and happy emoticons
emoticons = emoticons_happy.union(emoticons_sad)


#--------------------------------------------------------------------------------------------------------------------
## 2) Cleaning function

from nltk import wordpunct_tokenize
from nltk.corpus import stopwords
import nltk

#Method for clean tweets
def clean_tweets(tweet):
    stop_words = set(stopwords.words('english'))
 
    #Removing RT, Hashtags, URLs and any other undesired character
    ##print('Tweet: ' + tweet) #
    
    tweet = re.sub(r':', '', tweet)
    tweet = re.sub(r'‚Ä¶', '', tweet)
    tweet = re.sub(r"@[A-Za-z0-9_]+","",tweet, flags=re.MULTILINE)
    tweet = re.sub(r"(https|http)\S+", "", tweet)
    tweet = re.sub(r"[0-9]*", "", tweet)
    tweet = re.sub(r"(”|“|-|\+|`|#|,|'|&amp|‘|’|~|¬|;|)*", "", tweet)
    tweet = re.sub(r"(\v|\t|\f|)*", "", tweet)
    tweet = re.sub(r'(\â€”|\b|\u2014|\u2013|\u2122|\u2022|\u0001|\u2610|\u25A2|\u25AF|\u25FB|\u2026|/)', " ", tweet)
    tweet = re.sub(r'"', "", tweet)
    tweet = re.sub(r' [^a-z0-9]+ ', ' ', tweet, re.UNICODE|re.MULTILINE)
    tweet.encode('ascii', errors='ignore')
    tweet = re.sub(r"'", " ", tweet)
    tweet = tweet.lower()    
    
    #remove emojis from tweet
    tweet = emoji_pattern.sub(r'', tweet)
    ##print('Filtered tweet: ' + tweet) #
    word_tokens = word_tokenize(tweet)
 
    #filter using NLTK library append it to a string
    filtered_tweet = []
 
    #looping through conditions
    for w in word_tokens:
        #check tokens against stop words, emoticons and punctuations
        if w not in stop_words and w not in emoticons and w not in string.punctuation:
            ##print('Word: ' + w) #
            filtered_tweet.append(w)
    return ' '.join(filtered_tweet)

#--------------------------------------------------------------------------------------------------------------------

## 3) Writting function

from nltk import word_tokenize, FreqDist
#Method write_tweets()
def write_tweets(keyword, file, wopc):
    # If the file exists, then read the existing data from the CSV file.
    if os.path.exists(file):
        df = pd.read_csv(file, header=0)
    else:
        df = pd.DataFrame(columns=COLS)
    #page attribute in tweepy.cursor and iteration
    for page in tweepy.Cursor(api.search, q=keyword,
                              count=200, include_rts=False, since=start_date, tweet_mode="extended").pages(5): #tweet_mode = "compat"
        for status in page:
            new_entry = []
            status = status._json
 
            ## check whether the tweet is in english or skip to the next tweet
            if status['lang'] != 'en':
                continue
 
            #when run the code, below code replaces the retweet amount and
            #no of favorires that are changed since last download.
            if status['created_at'] in df['created_at'].values:
                i = df.loc[df['created_at'] == status['created_at']].index[0]
                if status['favorite_count'] != df.at[i, 'favorite_count'] or \
                   status['retweet_count'] != df.at[i, 'retweet_count']:
                    df.at[i, 'favorite_count'] = status['favorite_count']
                    df.at[i, 'retweet_count'] = status['retweet_count']
                continue

 
            #call clean_tweet method for extra preprocessing
            filtered_tweet = clean_tweets(status['full_text']) #text
 
            #pass textBlob method for sentiment calculations
            blob = TextBlob(filtered_tweet)
        
            Sentiment = blob.sentiment
 
            #seperate polarity and subjectivity in to two variables
            polarity = Sentiment.polarity
            subjectivity = Sentiment.subjectivity
 
            #new entry append
            new_entry += [status['id'], status['created_at'],
                          status['source'], status['full_text'],filtered_tweet, Sentiment,polarity,subjectivity, status['lang'],
                          status['favorite_count'], status['retweet_count']]
 
            #to append original author of the tweet
            new_entry.append(status['user']['screen_name'])
 
            try:
                is_sensitive = status['possibly_sensitive']
            except KeyError:
                is_sensitive = None
            new_entry.append(is_sensitive)
 
            # hashtagas and mentiones are saved using comma separted
            hashtags = ", ".join([hashtag_item['text'] for hashtag_item in status['entities']['hashtags']])
            new_entry.append(hashtags)
            mentions = ", ".join([mention['screen_name'] for mention in status['entities']['user_mentions']])
            new_entry.append(mentions)
 
            #get location of the tweet if possible
            try:
                location = status['user']['location']
            except TypeError:
                location = ''
            new_entry.append(location)
 
            try:
                coordinates = [coord for loc in status['place']['bounding_box']['coordinates'] for coord in loc]
            except TypeError:
                coordinates = None
            new_entry.append(coordinates)
            
            single_tweet_df = pd.DataFrame([new_entry], columns=COLS)
            df = df.append(single_tweet_df, ignore_index=True)
            csvFile = open(file, 'a' ,newline='', encoding='utf-8')
    if (wopc == 1):
        print("We are processing your file " + telemedicine_tweets + " ... ")
        df.to_csv(csvFile, mode='a', columns=COLS, index=False, encoding="utf-8")    
    #df.to_csv(csvFile, mode='a', columns=COLS, index=False, encoding="utf-8")


telemedicine_keywords = '#telemedicine OR #telehealth OR #digitalhealth OR #ehealth OR #digitalpatient OR #digitaltransformation'

#display_tweets(telemedicine_keywords)
write_tweets(telemedicine_keywords,  telemedicine_tweets, 1)

#--------------------------------------------------------------------------------------------------------------------
