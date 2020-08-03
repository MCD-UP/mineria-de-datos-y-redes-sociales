import json
from neo4j import GraphDatabase
import time

def store_tweet(tx, tweet):
    neo4j_params = {"user_id": tweet['user']['id'],
        "user_name": tweet['user']['name'],
        "tweet_id": tweet['id'],
        "tweet_text": tweet['text'],
        "tweet_time": time.strftime('%Y-%m-%d %H:%M:%S', time.strptime(tweet['created_at'],'%a %b %d %H:%M:%S +0000 %Y')),
        "mentions": tweet['entities']['user_mentions']
        }
    
    tx.run("""
        MERGE (u:User {uid: $user_id}) on create set u.name = $user_name
        MERGE (t:Tweet {uid: $tweet_id}) on create set t.text = $tweet_text, t.time = $tweet_time
        MERGE (u)-[:TWEETS]->(t)
        WITH t, $mentions as mentions 
        unwind mentions as mention
       MERGE (u:User {uid: mention.id}) on create set u.name = mention.name
       MERGE (t)-[:MENTIONS]->(u)
       """, neo4j_params)

def process_file(file_name):
    with GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "1234")) as driver:
        with open(file_name, 'r') as the_file:
            with driver.session() as session:
                with session.begin_transaction() as tx:
                    for line in the_file:
                        tweet = json.loads(line)
                        store_tweet(tx, tweet)

                        if 'retweeted_status' in tweet:
                            store_tweet(tx, tweet['retweeted_status'])
                            retweet_data = {
                            'tweet_id': tweet['id'],
                            "retweet_id": tweet['retweeted_status']['id']
                            }
                            
                            tx.run("""
                            MATCH (t:Tweet {uid: $tweet_id})
                            MATCH (r:Tweet {uid: $retweet_id})
                            MERGE (t)-[:RE_TWEETS]->(r)
                            """, retweet_data)                        

process_file('tweetsSmall.json')