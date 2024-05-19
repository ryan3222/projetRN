import numpy as np
import matplotlib.pyplot as plt 
import math
import sklearn as sk


import re
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import csv
import os


print(os.getcwd())
class TweetCleaner:
    def __init__(self):
        self.tokenizer = RegexpTokenizer(r'\w+')
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
     
    def clean_tweet(self,tweet):

        # Convertir en minuscules
        tweet = tweet.lower()
        
        # Supprimer les balises HTML
        tweet = re.sub(r'<[^>]+>', '', tweet)

        # Suprimer les @user
        tweet = re.sub(r'@\w*(?=\s)', '', tweet)

        # Suprimer les lien https
        tweet = re.sub(r'http(s?)://[\w\/\.]+(?=\s?)', '', tweet)

        # Suprimer les hashtags
        tweet = re.sub(r'#\w+', '', tweet)

        # Tokenisation des mots
        tokenizer = RegexpTokenizer(r'\w+')
        words = tokenizer.tokenize(tweet)

        # Supprimer stop words
        stop_words = set(stopwords.words('english'))
        words = [word for word in words if not word in stop_words]
        
        # Radicalisation des mots
        
        wnl = WordNetLemmatizer()
        words = [wnl.lemmatize(word) for word in words]

        
        # Supprimer les non-mots et la ponctuation
        tweet = ' '.join(re.findall(r'\b[a-z]+\b', ' '.join(words)))
        
        # Remplacer les espaces blancs multiples par un seul espace
        tweet = re.sub(r'\s+', ' ', tweet)
        
        return tweet

    def clean_csv(self,input_file, output_file):
        with open(input_file, 'r', encoding='utf-8') as csv_file:
            reader = csv.reader(csv_file)
            with open(output_file, 'w', newline='', encoding='utf-8') as cleaned_file:
                writer = csv.writer(cleaned_file)
                for row in reader:
                    cleaned_row = [row[0], row[1], row[2], row[3], row[4],  self.clean_tweet(row[5])]
                    writer.writerow(cleaned_row)






input_file = 'data.csv'
output_file = 'cleaned_data.csv'

try:
    os.remove(output_file)
    print("removed old " + output_file)
except OSError:
    pass
tweet_cleaner = TweetCleaner()
#tweet_cleaner.clean_csv(input_file, output_file)