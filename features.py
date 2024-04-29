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


def binary_feature_representation_for_data(data, vocabulary,output_file,indices_file):
    feature_vectors = []
    #word_to_index = {word: index for index, word in enumerate(vocabulary)}
    with open(output_file, 'w', newline='', encoding='utf-8') as output_csv:
        writer = csv.writer(output_csv)
        with open(data, 'r', encoding='utf-8') as data_csv, open(indices_file, 'r', encoding='utf-8') as indices_file:
            data_reader = csv.reader(data_csv)
            indices_reader = csv.reader(indices_file)
            for data_row, indices_row in zip(data_reader, indices_reader):
                feature_vector = [data_row[0]] #ECRIRE LE Y DANS LA PREMIERE COLONE
                feature_vector += [1 if str(index) in indices_row else 0 for index in range(len(vocabulary))]
                writer.writerow(feature_vector)

    return feature_vectors

# Exemple d'utilisation :
# Supposons que "data" est une liste contenant tous vos tweets prétraités
# et "vocabulary" est votre liste de mots de vocabulaire



cleaned_data ='cleaned_data.csv'
indices='indices.txt'
with open('vocab.txt', 'r', encoding='utf-8') as file:
        vocabulary = [line.strip() for line in file.readlines()]
print(vocabulary)
binary_feature_vectors = binary_feature_representation_for_data(cleaned_data, vocabulary,"final_data.csv",indices)
