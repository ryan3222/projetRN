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


def generer_vocab(cleaned_data, k):
    word_count = {}
    with open(cleaned_data, 'r', encoding='utf-8') as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            tweet=row[5]
            for word in tweet.split():
                word_count[word] = word_count.get(word, 0) + 1
    
    vocabulary = [word for word, count in word_count.items() if count >= k]
    return vocabulary


def map_words_to_indices(cleaned_data, vocabulary):
    indices = []
    with open(cleaned_data, 'r', encoding='utf-8') as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            tweet=row[5]
            for word in tweet.split():
                if word in vocabulary:
                    indices.append(vocabulary.index(word))
    return indices


k = 5  # Exemple de seuil d'occurrence
cleaned_data ='cleaned_data.csv'
vocabulary = generer_vocab(cleaned_data, k)
#sauvgarded le vocab
with open("vocab2.txt", 'w', encoding='utf-8') as file:
        for word in vocabulary:
            file.write(word + '\n')

# Mapper chaque mot du tweet prétraité à son index dans la liste de vocabulaire
indices_of_tweets = map_words_to_indices(cleaned_data, vocabulary)
print(vocabulary)