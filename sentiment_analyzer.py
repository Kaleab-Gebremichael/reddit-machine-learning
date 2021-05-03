from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

new_words = {
	'calls' : 
	'puts' : 
	'moon' : 
	'dd' : 
	'diligence' : 
	'gains' : 
	'yolo' : 
	'short' : 
	'long' : 
	'buy' : 
	'sell' : 
	'bull' : 
	'bear' : 
	'squeeze' : 
	
}

analyzer.lexicon.update(new_words)

#print(analyzer.lexicon)
