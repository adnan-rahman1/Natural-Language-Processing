import numpy as np
import pandas as pd
import nltk, re, string


data = pd.read_csv('SMSSpamCollection.tsv', sep='\t', header=None)
data.columns = ['label', 'body_text']


print(data.head())

stopword = nltk.corpus.stopwords.words('english')
def body_text_clean(text):
    text_nopunc = ''.join([ch for ch in text if ch not in string.punctuation])
    tokens = re.split('\W+', text)
    text = [word for word in tokens if word not in stopword]
    return text

data['body_text_clean'] = data['body_text'].apply(lambda x: body_text_clean(x.lower()))


wn = nltk.WordNetLemmatizer()

def text_lemmatizing(tokenize_text):
    lemmatize_text = [ wn.lemmatize(word) for word in tokenize_text ]
    return lemmatize_text

data['body_lemmatize_text'] = data['body_text_clean'].apply(lambda x: text_lemmatizing(x))

print(data.head())
