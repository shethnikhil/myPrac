from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
#note: depending on how you installed (e.g., using source code download versus pip install), you may need to import like this:
#from vaderSentiment import SentimentIntensityAnalyzer

# --- examples -------
sentences = ["congress party will never win",  # positive sentence example
             "Modi is a hero",  # punctuation emphasis handled correctly (sentiment intensity adjusted)
             "BJP will win elections" # booster words handled correctly (sentiment intensity adjusted
             ]

analyzer = SentimentIntensityAnalyzer()
for sentence in sentences:
    vs = analyzer.polarity_scores(sentence)
    print("{:-<65} {}".format(sentence, str(vs)))