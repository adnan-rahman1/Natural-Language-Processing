import numpy as np
import pandas as pd
import nltk, re, string


data = pd.read_csv('SMSSpamCollection.tsv', sep='\t', header=None)
data.columns = ['label', 'body_text']


stopword = nltk.corpus.stopwords.words('english')
def body_text_clean(text):
    text_nopunc = ''.join([ch for ch in text if ch not in string.punctuation])
    tokens = re.split('\W+', text)
    text = [word for word in tokens if word not in stopword]
    return text

data['body_text_clean'] = data['body_text'].apply(lambda x: body_text_clean(x.lower()))


ps = nltk.PorterStemmer()

def text_stemming(tokenize_text):
    stem_text = [ ps.stem(word) for word in tokenize_text ]
    return stem_text

data['body_stem_text'] = data['body_text_clean'].apply(lambda x: text_stemming(x))

print(data.head())
