from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from string import punctuation
import nltk
import string
from collections import defaultdict
from heapq import nlargest
import operator

text = ("Srei Infrastructure Finance (Srei) reassured stakeholders about its financial health and said the current market development should not be linked to the company's growth.The recent market developments in reference to non-bank financial institutions (NBFCs) should not be considered while evaluating its stable financial health and prudent growth strategies, it said in a statement.Srei emphasised that it has been timely meeting its liability commitments to banks, financial institutions and investors.Srei Infra, including Srei Equipment Finance, has repaid all its debt obligations as on date without any delay and has enough liquidity to honour all its financial obligations, it added.")
# Create sentences
sentences = sent_tokenize(text)

# Set stop-words
stopwords = set(stopwords.words('german') + stopwords.words('english') + list(punctuation))

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.wordnet import WordNetLemmatizer

def tokenize(text):
    tokens = nltk.word_tokenize(text)
    stems = []
    for item in tokens:
        stems.append(WordNetLemmatizer().lemmatize(item))
    return stems

tfidf = TfidfVectorizer(tokenizer=tokenize, 
                        stop_words=stopwords)
tfs = tfidf.fit_transform([text])

# Frequencies
freqs = {}

feature_names = tfidf.get_feature_names()
for col in tfs.nonzero()[1]:
    freqs[feature_names[col]] = tfs[0, col]
    
important_sentences = defaultdict(int)

for i, sentence in enumerate(sentences):
    for token in word_tokenize(sentence.lower()):
        if token in freqs:
            important_sentences[i] += freqs[token]
            

# Choose 20% of the text to show
number_sentences = int(len(sentences) * 0.2)

# Create an index with the most important sentences
index_important_sentences = nlargest(number_sentences, 
                                   important_sentences, 
                                   important_sentences.get)

# Sort frequencies
sorted_freqs = sorted(freqs.items(), key=operator.itemgetter(1), reverse=True)

# Show important words
print('5 most important words:\n')

for i in range(5):
    print('-',sorted_freqs[i][0])
    
# Create summary
print('\nSumary:\n')
for i in sorted(index_important_sentences):
    print(sentences[i]+'\n')            