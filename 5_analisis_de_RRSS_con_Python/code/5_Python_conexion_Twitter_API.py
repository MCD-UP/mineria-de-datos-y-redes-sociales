import tweepy
import time
import json
 
# consumer keys and access tokens, used for OAuth
consumer_key="XXX"
consumer_secret="XXXX"
access_token="XXX"
access_token_secret="XXX"
 
# OAuth process, using the keys and tokens
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
 
# creation of the actual interface, using authentication
api = tweepy.API(auth)

user = api.me() 
 
print('Name: ' + user.name)
print('Location: ' + user.location)
print('Friends: ' + str(user.friends_count))

print('\n\n')
search = tweepy.Cursor(api.search, q="Mexico", result_type="recent").items(5)

#Mostrar los resultados recuperados
for item in search: 
    print (item.text)

    
def limit_handled(cursor):
    while True:
        try:
            yield cursor.next()
        except tweepy.TweepError as e:
            #print(e.error_msg)
            print('You are unable to follow more people at this time.') 
            time.sleep(15 * 60)
            exit()

def search(keyword):
# Extract the first "xxx" tweets related to "fast car"
    with open('tweetsSearch.json', 'a') as the_file:
        for tweet in limit_handled(tweepy.Cursor(api.search, q=keyword, since='2017-05-09').items(15)):
            the_file.write(json.dumps(tweet._json) + '\n')
            
#
search("Science")