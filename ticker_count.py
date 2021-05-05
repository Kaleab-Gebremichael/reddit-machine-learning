import couchdb
import json
import pandas as pd
from get_tickers import tickers

couch = couchdb.Server('http://admin:password@129.114.27.202:30003/')
db = couch['reddit_data']

x_arr = []
y_arr = []

for row in db.view('_all_docs', include_docs=True):
	x = json.dumps(row['doc']['body'])
	y = json.dumps(row['doc']['sentiment'])
	x_arr.append(x)
	y_arr.append(y)

data = pd.DataFrame(list(zip(x_arr,y_arr)))
data.columns = ['body', 'sentiment']
data['sentiment'] = data['sentiment'].apply(lambda x: float(x))	#turn into floating points

output = pd.DataFrame(columns=['ticker', 'count', 'sentiment'])

for i in range(len(tickers)):
	ticker = tickers[i]
	a = data['body'].str.contains(" " + ticker + " ", case=False).sum()
	b = data[(data['body'].str.contains(" " + ticker + " ", case=False)) & (data['sentiment'] != 0)]['sentiment']
	if (b.empty):
		continue
	output.loc[i] = [ticker, a, b.mean()]


print(output.sort_values('count')[::-1])

