from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

new_words = {
	'calls' : 1,
	'puts' : -2,
	'moon' : 1,
	'dd' : 2,
	'diligence' : 2,
	'gains' : 1,
	'yolo' : 1,
	'short' : -2,
	'long' : 1,
	'buy' : 1,
	'sell' : -2,
	'bull' : 1,
	'bear' : -2,
	'squeeze' : 1.5,
	'correction' : -3
}

analyzer.lexicon.update(new_words)

#print(analyzer.lexicon)
