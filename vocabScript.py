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
    word_to_index = {word: index for index, word in enumerate(vocabulary)}
    indices = []
    with open(cleaned_data, 'r', encoding='utf-8') as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            tweet = row[5]
            tweet_indices_row = []
            for word in tweet.split():
                if word in word_to_index:
                     tweet_indices_row.append(word_to_index[word])
            indices.append(tweet_indices_row)        
    return indices

<<<<<<< Updated upstream
k = 9000  # Exemple de seuil d'occurrence 3200 pour avoir 600
=======
k = 15000  # Exemple de seuil d'occurrence 3200 pour avoir 600
>>>>>>> Stashed changes
cleaned_data ='cleaned_data.csv'
vocabulary = generer_vocab(cleaned_data, k)

#sauvgarded le vocab
with open("vocab.txt", 'w', encoding='utf-8') as file:
        for word in vocabulary:
            file.write(word + '\n')

# Mapper chaque mot du tweet prétraité à son index dans la liste de vocabulaire
indices_of_tweets = map_words_to_indices(cleaned_data, vocabulary)
def write_indices_to_file(tweet_indices, output_file):
    with open(output_file, 'w', encoding='utf-8') as file:
        for indices_row in tweet_indices:
            file.write(','.join(map(str, indices_row)) + '\n')

write_indices_to_file(indices_of_tweets, 'indices.txt')