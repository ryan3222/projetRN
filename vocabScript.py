import numpy as np
import matplotlib.pyplot as plt 
import math
import sklearn as sk


import re
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import csv
import os



def build_vocabulary(cleaned_data, k):
    word_count = {}
    with open(cleaned_data, 'r', encoding='utf-8') as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            tweet=row[5]
            for word in tweet.split():
                word_count[word] = word_count.get(word, 0) + 1
    
    # Filtrer les mots qui apparaissent au moins min_occurrences fois
    vocabulary = [word for word, count in word_count.items() if count >= k]
    return vocabulary

def save_vocabulary(vocabulary, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        for word in vocabulary:
            file.write(word + '\n')

def map_words_to_indices(tweet, vocabulary):
    indices = []
    for word in tweet.split():
        if word in vocabulary:
            indices.append(vocabulary.index(word))
    return indices

# Exemple d'utilisation
preprocessed_tweets = ["tweet prétraité 1", "tweet prétraité 2", ...]  # Vos tweets prétraités
k = 5  # Exemple de seuil d'occurrence
vocabulary = build_vocabulary(preprocessed_tweets, k)
save_vocabulary(vocabulary, 'vocab.txt')

# Mapper chaque mot du tweet prétraité à son index dans la liste de vocabulaire
indices_of_tweets = [map_words_to_indices(tweet, vocabulary) for tweet in preprocessed_tweets]