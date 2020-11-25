import re
import os
import sys
import json

import pandas as pd
import numpy as np
import spacy

from bs4 import BeautifulSoup
import unicodedata
from textblob import TextBlob

import nltk; nltk.download('stopwords')
# NLTK Stop words
from nltk.corpus import stopwords
russian_stopwords = stopwords.words("russian")

from pymorphy2 import MorphAnalyzer

from sklearn.feature_extraction.text import CountVectorizer


#path = os.path.dirname(os.path.abspath(__file__))
#abbreviations_path = os.path.join(path, 'data','abbreviations_wordlist.json')


def get_wordcounts(x):
    length = len(str(x).split())
    return length

def get_charcounts(x):
    s = x.split()
    x = ''.join(s)
    return len(x)

def get_avg_wordlength(x):
    count = _get_charcounts(x)/_get_wordcounts(x)
    return count

def get_stopwords_counts(x):
    l = len([t for t in x.split() if t in russian_stopwords])
    return l

def get_hashtag_counts(x):
    l = len([t for t in x.split() if t.startswith('#')])
    return l

def get_mentions_counts(x):
    l = len([t for t in x.split() if t.startswith('@')])
    return l

def get_digit_counts(x):
    digits = re.findall(r'[0-9,.]+', x)
    return len(digits)

def get_uppercase_counts(x):
    return len([t for t in x.split() if t.isupper()])

"""
def cont_exp(x):
    abbreviations = json.load(open(abbreviations_path))

    if type(x) is str:
        for key in abbreviations:
            value = abbreviations[key]
            x = x.replace(key, value)
        return x
    else:
        return x
"""

def get_emails(x):
    emails = re.findall(r'([a-z0-9+._-]+@[a-z0-9+._-]+\.[a-z0-9+_-]+\b)', x)
    counts = len(emails)

    return counts, emails


def remove_emails(x):
    return re.sub(r'([a-z0-9+._-]+@[a-z0-9+._-]+\.[a-z0-9+_-]+)',"", x)

def get_urls(x):
    urls = re.findall(r'(http|https|ftp|ssh)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?', x)
    counts = len(urls)

    return counts, urls

def remove_urls(x):
    return re.sub(r'(http|https|ftp|ssh)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?', '' , x)

def remove_rt(x):
    return re.sub(r'\brt\b', '', x).strip()

def remove_special_chars(x):
    x = re.sub(r'[^\w ]+', "", x)
    x = ' '.join(x.split())
    return x

def remove_html_tags(x):
    return BeautifulSoup(x, 'lxml').get_text().strip()

def remove_accented_chars(x):
    x = unicodedata.normalize('NFKD', x).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return x

def remove_stopwords(x):
    return ' '.join([t for t in x.split() if t not in stopwords])	

"""
def make_base(x):
    x = str(x)
    x_list = []
    doc = nlp(x)

    for token in doc:
        lemma = token.lemma_
        if lemma == '-PRON-' or lemma == 'be':
            lemma = token.text

        x_list.append(lemma)
    return ' '.join(x_list)
"""

def make_base(x):
    all_word_str = " ".join(x)
    all_word_list = all_word_str.split()
    all_unique_word = pd.Series(all_word_list).unique()
    lemmatized_word_dict = {}
    lemmatizer = MorphAnalyzer()
    for word in all_unique_word:
        lemmatized_word_dict[word] = lemmatizer.normal_forms(word)[0]
    x_list = ' '.join([lemmatized_word_dict[word] for word in text])
    return x_list

def get_value_counts(df, col):
    text = ' '.join(df[col])
    text = text.split()
    freq = pd.Series(text).value_counts()
    return freq

def remove_common_words(x, freq, n=20):
    fn = freq[:n]
    x = ' '.join([t for t in x.split() if t not in fn])
    return x

def remove_rarewords(x, freq, n=20):
    fn = freq.tail(n)
    x = ' '.join([t for t in x.split() if t not in fn])
    return x

def remove_dups_char(x):
    x = re.sub("(.)\\1{2,}", "\\1", x)
    return x

def spelling_correction(x):
    x = TextBlob(x).correct()
    return x

def get_basic_features(df):
    if type(df) == pd.core.frame.DataFrame:
        df['char_counts'] = df['text'].apply(lambda x: get_charcounts(x))
        df['word_counts'] = df['text'].apply(lambda x: get_wordcounts(x))
        df['avg_wordlength'] = df['text'].apply(lambda x: get_avg_wordlength(x))
        df['stopwords_counts'] = df['text'].apply(lambda x: get_stopwords_counts(x))
        df['hashtag_counts'] = df['text'].apply(lambda x: get_hashtag_counts(x))
        df['mentions_counts'] = df['text'].apply(lambda x: get_mentions_counts(x))
        df['digits_counts'] = df['text'].apply(lambda x: get_digit_counts(x))
        df['uppercase_counts'] = df['text'].apply(lambda x: get_uppercase_counts(x))
    else:
        print('ERROR: This function takes only Pandas DataFrame')
        
    return df


def get_ngram(df, col, ngram_range):
    vectorizer = CountVectorizer(ngram_range=(ngram_range, ngram_range))
    vectorizer.fit_transform(df[col])
    ngram = vectorizer.vocabulary_
    ngram = sorted(ngram.items(), key = lambda x: x[1], reverse=True)

    return ngram


