import numpy as np
import pandas as pd
import nltk, re, string


data = pd.read_csv('SMSSpamCollection.tsv', sep='\t', header=None)
data.columns = ['label', 'body_text']

def remove_punc(text):
    text_nopunc = ''.join([ch for ch in text if ch not in string.punctuation])
    return text_nopunc


data['body_text_clean'] = data['body_text'].apply(lambda x: remove_punc(x))


def tokenize(text):
    tokens = re.split('\W+', text)
    return tokens


data['body_text_tokenize'] = data['body_text_clean'].apply(lambda x: tokenize(x.lower()))


# remove most commonly used words
stopword = nltk.corpus.stopwords.words('english')
def remove_stopwords(tokenized_list):
    text = [word for word in tokenized_list if word not in stopword]
    return text

data['body_text_nostopwords'] = data['body_text_tokenize'].apply(lambda x: remove_stopwords(x))

print(data.head(3))
