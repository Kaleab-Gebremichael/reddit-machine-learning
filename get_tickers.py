import json
import urllib.request

NYSE_LISTED = 'https://pkgstore.datahub.io/core/nyse-other-listings/nyse-listed_json/data/e8ad01974d4110e790b227dc1541b193/nyse-listed_json.json'
OTHER_LISTED = 'https://pkgstore.datahub.io/core/nyse-other-listings/other-listed_json/data/e95106d7c30d265a719c5ff43843907a/other-listed_json.json'

with urllib.request.urlopen(NYSE_LISTED) as url:
	nyse_data = json.loads(url.read().decode())

with urllib.request.urlopen(OTHER_LISTED) as url:
	other_data = json.loads(url.read().decode())

tickers = []
for company in nyse_data:
	tickers.append(company['ACT Symbol'])

for company in other_data:
	tickers.append(company['ACT Symbol'])
