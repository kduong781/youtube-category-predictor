#! /bin/python

import argparse
import pickle
import json
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import string
import pandas as pd
import numpy as np
import re

def clean(title):
    '''
        Cleans the strings in specified column
    '''
    table = str.maketrans('', '', string.punctuation)
    stop_words = nltk.corpus.stopwords.words('english')
    porter = nltk.stem.porter.PorterStemmer()

    line = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', title)
    tokens = nltk.word_tokenize(line)
    tokens = [word.lower() for word in tokens]
    stripped = [w.translate(table) for w in tokens]
    words = [word for word in stripped if word.isalpha()]
    words = [w for w in words if not w in stop_words]
    stemmed = [porter.stem(word) for word in words]
    stemmed = [word.strip() for word in stemmed if len(word) > 3]

    return ' '.join(stemmed)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("title", help="title of youtbe video to predict")
    parser.add_argument("description", help="description of youtbe video to predict")
    parser.add_argument("-v", "--verbose", default=False, action='store_true', help="adds more information to output")
    args = parser.parse_args()

    model = None
    tfidf_title = None
    tfidf_description = None
    tfidf_tags = None
    categories = None
    labels = [None] * 32

    with open('data/linearSVC.pkl', 'rb') as f:
        model = pickle.load(f)

    with open('data/tfidf_title.pkl', 'rb') as f:
        tfidf_title = pickle.load(f)

    with open('data/tfidf_description.pkl', 'rb') as f:
        tfidf_description = pickle.load(f)

    with open('data/tfidf_tags.pkl', 'rb') as f:
       tfidf_tags = pickle.load(f)

    with open('data/US_category_id.json', 'r') as f:
        categories = json.load(f)

    for obj in categories['items']:
        labels.insert(int(obj['id']), obj['snippet']['title'])

    if not model or not tfidf_title or not tfidf_description or not tfidf_tags or not categories:
        print('pickles did not successfully load. Please ensure the .pkl and json files are located in ./data/')

    clean_title = [clean(args.title.strip())]
    clean_description = [clean(args.description.strip())]
    clean_tags = [clean('')]

    vect_title = tfidf_title.transform(clean_title).toarray()
    vect_description = tfidf_description.transform(clean_description).toarray()
    vect_tags = tfidf_tags.transform(clean_tags).toarray()
    sample = np.concatenate([vect_title, vect_description, vect_tags], axis=1)

    if args.verbose:
        print(sample.shape)
    
    pred = model.predict(sample)[0]
    if args.verbose:
        print(pred)

    print(labels[pred])

if __name__ == '__main__':
    main()