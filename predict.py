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
import requests
import sys

def setup(api_path):
    '''
        Get api key
    '''
    with open(api_path, 'r') as file:
        api_key = file.readline()

    return api_key

def api_request(video_id, api_key, country_code):
    '''
        Requests info on a youtube video
    '''
    request_url = f"https://youtube.googleapis.com/youtube/v3/videos?part=snippet%2CcontentDetails%2Cstatistics&id={video_id}&regionCode={country_code}&key={api_key}"
    request = requests.get(request_url)
    if request.status_code == 429:
        print("Temp-Banned due to excess requests, please wait and continue later")
        sys.exit()
    return request.json()

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
    parser.add_argument("video_id", help="id of youtube video to predict")
    parser.add_argument("country_code", help="region that the video is from")
    # parser.add_argument("title", help="title of youtbe video to predict")
    # parser.add_argument("description", help="description of youtbe video to predict")
    parser.add_argument("-v", "--verbose", default=False, action='store_true', help="adds more information to output")
    args = parser.parse_args()

    api_key = setup('apikey')
    api_response = api_request(args.video_id, api_key, args.country_code)
    target_title = None
    target_description = None
    target_tags = None
    target_category_id = None

    try:
        video = api_response.get("items", None)[0] # might be an empty array
        target_title = video["snippet"]["title"]
        target_description = video["snippet"]["description"]
        target_tags = ' '.join(video["tags"])
        target_category_id = video["categoryId"]
    except Exception:
        print(f'No results found for\nvideo_id:{args.video_id}\ncountry_code:{args.country_code}\n')
        sys.exit()

    model = None
    tfidf_title = None
    tfidf_description = None
    tfidf_tags = None
    categories = None
    labels = [None] * 45

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

    clean_title = [clean(target_title)]
    clean_description = [clean(target_description)]
    clean_tags = [clean(target_tags)]
    '''
    clean_title = [clean(args.title.strip())]
    clean_description = [clean(args.description.strip())]
    clean_tags = [clean('')]
    '''

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

    print(f'Target category: {labels[int(target_category_id)]}')

if __name__ == '__main__':
    main()
