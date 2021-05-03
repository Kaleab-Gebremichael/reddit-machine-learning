import praw
import os
import time
import couchdb
import json
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from unidecode import unidecode


couch = couchdb.Server('http://admin:password@129.114.27.202:30003/')
try:
    db = couch.create('reddit_data')
except:
    db = couch['reddit_data']

reddit = praw.Reddit(
	client_id=os.environ.get('CLIENT_ID'),
	client_secret=os.environ.get('CLIENT_SECRET'),
	user_agent='my user agent'
)


analyzer = SentimentIntensityAnalyzer()

while True:
	try:
		subreddit = reddit.subreddit("wallstreetbets")
		for comment in subreddit.stream.comments(skip_existing=True):
			cur_time = time.time()
			subreddit = str(comment.subreddit)
			title = str(comment.link_title)
			body = str(comment.body)[:3000]	#at most 3000 characters
			vs = analyzer.polarity_scores(unidecode(body))
			sentiment = vs['compound']	# want the compound score

			output = {
				'date' : cur_time,
				'subreddit' : subreddit,
				'title' : title,
				'body' : body,
				'sentiment' : sentiment
			}

			db.save(output)

	except Exception as e:
		print(str(e))
		time.sleep(10)
