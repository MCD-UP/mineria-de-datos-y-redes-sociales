## 4) Tokenizing from CSV file

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
import re
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score

alltweets = pd.read_csv('telemedicine_data4.csv') #csvFile

alltweets

#--------------------------------------------------------------------------------------------------------------------

#Data partitioning

from numpy.random import RandomState
import pandas as pd

rng = RandomState()

train_tweets = alltweets.sample(frac=0.7, random_state=rng)
test_tweets = alltweets.loc[~alltweets.index.isin(train_tweets.index)]

train_tweets = train_tweets[['polarity','clean_text']]
test = test_tweets['clean_text']

#--------------------------------------------------------------------------------------------------------------------

#Tokenizing

pd.options.display.max_colwidth = max (train_tweets.loc[:,'clean_text'].size, test_tweets.loc[:,'clean_text'].size)

import nltk
from nltk.corpus import sentiwordnet as swn
from nltk.tag import pos_tag, map_tag

def tokenization(tweet):
        tweet = str(tweet)
        tweet = re.sub(r"[\u2026|00-9]*", "", tweet)
        tweet = word_tokenize(tweet) #
        return [word for word in tweet]
    
def sentimentFromPolarity(tweet_polarity):
        if (tweet_polarity > 0):
            return 'positive'
        else:
            return 'negative'

train_tweets.loc[:,'tweet_list'] = train_tweets.loc[:,'clean_text'].apply(tokenization)
test_tweets.loc[:,'tweet_list']  = test_tweets.loc[:,'clean_text'].apply(tokenization)

train_tweets.loc[:,'sentiment'] = train_tweets.loc[:,'polarity'].apply(sentimentFromPolarity)
test_tweets.loc[:,'sentiment']  = test_tweets.loc[:,'polarity'].apply(sentimentFromPolarity)

train_tweets

#--------------------------------------------------------------------------------------------------------------------

#Tokenizing

trainingSet_WS = []
testinggSet_WS = []

for clean_text, sentiment in zip(train_tweets.tweet_list, train_tweets.sentiment):  
    trainingSet_WS.append((clean_text, sentiment))
    #print(phrasedocs)
    
for clean_text, sentiment in zip(test_tweets.tweet_list, test_tweets.sentiment):  
    testinggSet_WS.append((clean_text, sentiment))

testinggSet_WS

#--------------------------------------------------------------------------------------------------------------------

#ML analysis

import nltk 

def buildVocabulary(trainingData):
    all_words = []
    
    for (words, sentiment) in trainingData:
        all_words.extend(words)

    wordlist = nltk.FreqDist(all_words)
    word_features = wordlist.keys()
    
    return word_features


def extract_features(tweet):
    tweet_words = set(tweet)
    features = {}
    for word in word_features:
        features['contains(%s)' % word] = (word in tweet_words)
    return features 

word_features = buildVocabulary(trainingSet_WS)
trainingFeatures = nltk.classify.apply_features(extract_features, trainingSet_WS)

#--------------------------------------------------------------------------------------------------------------------

#ML analysis - Training by NaiveBayesClassifier

start_time = time.time()
classifier = nltk.NaiveBayesClassifier.train(trainingFeatures)
elapsed_time = time.time() - start_time
print("Run time:",elapsed_time/60/60,"hours")

print("Relation between a word and its sentiment")
print(classifier.show_most_informative_features(100))

#--------------------------------------------------------------------------------------------------------------------

#ML analysis - Testing

import nltk 

NBResultLabels = [classifier.classify(extract_features(tweet[0])) for tweet in testinggSet_WS]
pCount = NBResultLabels.count('positive')
nCount = NBResultLabels.count('negative')

# get the majority vote
if pCount > nCount:
    print("Overall Positive Sentiment")
    print("Positive Sentiment Percentage = " + str(100*pCount/len(NBResultLabels)) + "%")
else: 
    print("Overall Negative Sentiment")
    print("Negative Sentiment Percentage = " + str(100*nCount/len(NBResultLabels)) + "%")


#--------------------------------------------------------------------------------------------------------------------

#Results and Visualization

print("Total training tweets: " + str(len(train_tweets)))
print("Positive tweets: " + str(train_tweets.sentiment.str.count("positive").sum()))
print("Negative tweets: " + str(train_tweets.sentiment.str.count("negative").sum()))

#--------------------------------------------------------------------------------------------------------------------

pos_text = []
neg_text = []

for clean_text, sentiment in zip(train_tweets.tweet_list, train_tweets.sentiment):  
    if(sentiment=='positive'):
        pos_text+=clean_text
    else:
        neg_text+=clean_text

pos_text

#--------------------------------------------------------------------------------------------------------------------

#WordCloud

from wordcloud import WordCloud 

pos_string=(" ").join(pos_text)
list_text = [pos_string]
for txt in list_text:
    word_cloud = WordCloud(width = 600,height = 600,max_font_size = 200).generate(txt)
    plt.figure(figsize=(12,10))# create a new figure
    plt.imshow(word_cloud,interpolation="bilinear")
    plt.axis("off")
    plt.show()



