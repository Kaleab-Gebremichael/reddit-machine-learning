import praw
import os
import time
import couchdb
import json
from unidecode import unidecode
from get_tickers import tickers
from sentiment_analyzer import analyzer

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

while True:
	try:
		subreddit = reddit.subreddit("wallstreetbets")
		for comment in subreddit.stream.comments(skip_existing=False):
			cur_time = time.time()
			subreddit = str(comment.subreddit)
			title = str(comment.link_title)
			body = str(comment.body)[:4000]	#at most 4000 characters

			if not any(x.lower() in body.split() for x in tickers):
				continue

			title_sentiment = analyzer.polarity_scores(unidecode(title.lower()))['compound']
			body_sentiment = analyzer.polarity_scores(unidecode(body.lower()))['compound']
			sentiment = title_sentiment + body_sentiment	# want the combined score

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
