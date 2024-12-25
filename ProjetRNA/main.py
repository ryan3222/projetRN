import sys
import os
import csv
import re
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import numpy as np
import tensorflow as tf
from features import TweetFeatureExtractor
from tweet_clean import TweetCleaner
from vocabScript import VocabularyGenerator

def main():
    vg = None  
    while True:
        print("Options:")
        print("1. Nettoyer le dataset de tweets")
        print("3. Générer un vocabulaire à partir du dataset nettoyé")
        print("4. Générer un vecteur de caractéristiques pour un seul tweet")
        print("5. Générer des vecteurs de caractéristiques pour l'ensemble du dataset")
        print("6. Prédire")
        print("7. Quitter")

        choice = input("Entrez votre choix : ")

        if choice == '1':
            input_file = input("Entrez le chemin du dataset d'entrée : ")
            output_file = input("Entrez le chemin du dataset nettoyé de sortie : ")
            tweet_cleaner = TweetCleaner()
            tweet_cleaner.clean_csv(input_file, output_file)
            print(f"Le dataset de tweets nettoyé est enregistré dans {output_file}")

        elif choice == '3':
            cleaned_data = input("Entrez le chemin du dataset nettoyé : ")
            k = int(input("Entrez le seuil de fréquence pour le vocabulaire : "))
            vg = VocabularyGenerator(cleaned_data, k)
            vocabulary = vg.generer_vocab()
            with open("vocab.txt", 'w', encoding='utf-8') as file:
                for word in vocabulary:
                    file.write(word + '\n')
            print("Le vocabulaire est enregistré dans vocab.txt")

        elif choice == '4':
            tweet = input("Entrez le tweet : ")
            with open('vocab.txt', 'r', encoding='utf-8') as file:
                vocabulary = [line.strip() for line in file.readlines()]
            feature_extractor = TweetFeatureExtractor(vocabulary)
            cleaned_tweet = TweetCleaner().clean_tweet(tweet)
            feature_vector = feature_extractor.binary_feature_representation_for_tweet(cleaned_tweet)
            print(f"Vecteur de caractéristiques pour le tweet : {feature_vector}")

        elif choice == '5':
            if vg is None:
                print("Veuillez d'abord générer le vocabulaire en utilisant l'option 3.")
                continue
            cleaned_data = input("Entrez le chemin du dataset nettoyé : ")
            output_file = input("Entrez le chemin du fichier de sortie des vecteurs de caractéristiques : ")
            with open('vocab.txt', 'r', encoding='utf-8') as file:
                vocabulary = [line.strip() for line in file.readlines()]

            indices_of_tweets = vg.map_words_to_indices()
            vg.write_indices_to_file(indices_of_tweets, 'indices.txt')
            feature_extractor = TweetFeatureExtractor(vocabulary)
            feature_extractor.binary_feature_representation_for_data(cleaned_data, output_file, 'indices.txt')
            print(f"Les vecteurs de caractéristiques pour le dataset sont enregistrés dans {output_file}")

        elif choice == '6':
            # load model
            model = tf.keras.models.load_model('models\DLClassifier-v001')

            # Read vocabulary from file
            with open('vocab.txt', 'r', encoding='utf-8') as file:
                vocabulary = [line.strip() for line in file.readlines()]

            # Initialize TweetCleaner and TweetFeatureExtractor
            tweet_cleaner = TweetCleaner()
            feature_extractor = TweetFeatureExtractor(vocabulary)

            tweet = input("\nEntrez le tweet : ")

            # Clean the tweet
            cleaned_tweet = tweet_cleaner.clean_tweet(tweet)
            print("\nCleaned tweet:", cleaned_tweet)

            # Generate feature vector for the tweet
            feature_vector = feature_extractor.binary_feature_representation_for_tweet(cleaned_tweet)
            print("\nFeature vector for the tweet:", feature_vector)

            # Predict
            pred = model.predict(np.array([feature_vector, ]))

            if(pred[0] > 0.5) :
                print("\n>>> this is a positive tweet!\n")
            else :
                print("\n>>> this is a negative tweet!\n")


        elif choice == '7':
            break

        else:
            print("Choix invalide. Veuillez réessayer.")

if __name__ == "__main__":
    main()
