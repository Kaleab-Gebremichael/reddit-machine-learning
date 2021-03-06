import json
import urllib.request

NYSE_LISTED = 'https://pkgstore.datahub.io/core/nyse-other-listings/nyse-listed_json/data/e8ad01974d4110e790b227dc1541b193/nyse-listed_json.json'
NASDAQ_LISTED = 'https://pkgstore.datahub.io/core/nasdaq-listings/nasdaq-listed-symbols_json/data/5c10087ff8d283899b99f1c126361fa7/nasdaq-listed-symbols_json.json'
OTHER_LISTED = 'https://pkgstore.datahub.io/core/nyse-other-listings/other-listed_json/data/e95106d7c30d265a719c5ff43843907a/other-listed_json.json'

#with urllib.request.urlopen(NYSE_LISTED) as url:
#	nyse_data = json.loads(url.read().decode())

#with urllib.request.urlopen(OTHER_LISTED) as url:
#	other_data = json.loads(url.read().decode())

with urllib.request.urlopen(NASDAQ_LISTED) as url:
        nasdaq_data = json.loads(url.read().decode())


tickers = []
#for company in nyse_data:
#	tickers.append(company['ACT Symbol'])

#for company in other_data:
#	tickers.append(company['ACT Symbol'])

for company in nasdaq_data:
	tickers.append(company['Symbol'])

tickers = list(set(tickers))	#remove duplicates

common_words = ["A", "AND", "I", "FOR", "IT", "ARE", "IF", "SO", "MY", "AT", \
		"ALL", "MORE", "DO", "OUT", "ONE", "HAS", "GOOD", "TIME", "AN",\
		 "NOW"]

for word in common_words:
	try:
		tickers.remove(word)
	except:
		pass
