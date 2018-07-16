import numpy as np
import pandas as pd
import nltk, re, string


data = pd.read_csv('SMSSpamCollection.tsv', sep='\t', header=None)
data.columns = ['label', 'body_text']

def remove_punc(text):
    text_nopunc = ''.join([ch for ch in text if ch not in string.punctuation])
    return text_nopunc


data['body_text_clean'] = data['body_text'].apply(lambda x: remove_punc(x))

print(data.head())

