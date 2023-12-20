#working example based off https://www.datacamp.com/tutorial/text-analytics-beginners-nltk

import pandas as pd
import csv
import nltk

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

df = pd.read_csv('https://raw.githubusercontent.com/pycaret/pycaret/master/datasets/amazon.csv');

def preprocess_text(text):
    # tokenize the text
    tokens = word_tokenize(text.lower())
    #remove stopwords
    filtered_tokens = [token for token in tokens if token not in stopwords.words('english')]
    #lemmatize the token
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    # Join the tokens back into a string
    processed_text = ' '.join(lemmatized_tokens)
    return processed_text

#apply the function to df
df['reviewText'] = df['reviewText'].apply(preprocess_text)

# initialize NLTK sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

#create get_sentiment function - can categorise as needed to create neutral field
def get_sentiment(text):
    scores = analyzer.polarity_scores(text)
    sentiment = 'positive' if scores['pos'] > 0 else 'negative'
    return sentiment

# apply get_sentiment function
df['sentiment'] = df['reviewText'].apply(get_sentiment)

print(df)