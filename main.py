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


print(os.getcwd())

def clean_tweet(tweet):

    # Convertir en minuscules
    tweet = tweet.lower()
    
    # Supprimer les balises HTML
    tweet = re.sub(r'<[^>]+>', '', tweet)

    # Suprimer les @

    tweet = re.sub(r'@\w*(?=\s)', '', tweet)

    # Suprimer les lien https
    tweet = re.sub(r'http(s?)://[\w\/\.]+(?=\s)', '', tweet)
    
    # Initialiser le stemmer
    stemmer = PorterStemmer()
    
    # Tokenisation des mots
    words = word_tokenize(tweet)
    
    # Supprimer les mots vides
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    
    # Radicalisation des mots
    words = [stemmer.stem(word) for word in words]
    
    # Supprimer les non-mots et la ponctuation
    tweet = ' '.join(re.findall(r'\b[a-zA-Z][a-z\']*\b', ' '.join(words)))
    
    # Remplacer les espaces blancs multiples par un seul espace
    tweet = re.sub(r'\s+', ' ', tweet)
    
    return tweet

def clean_csv(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as csv_file:
        reader = csv.reader(csv_file)
        with open(output_file, 'w', newline='', encoding='utf-8') as cleaned_file:
            writer = csv.writer(cleaned_file)
            for row in reader:
                cleaned_row = [row[0], row[1], row[2], row[3], row[4], clean_tweet(row[5])]
                writer.writerow(cleaned_row)

input_file = 'data.csv'
output_file = 'cleaned_data.csv'
clean_csv(input_file, output_file)