import sys

import numpy as np

from tweet_clean import TweetCleaner
from features import TweetFeatureExtractor

import tensorflow as tf

# load model
model = tf.keras.models.load_model('models\DLClassifier-v001')

# Read vocabulary from file
with open('vocab.txt', 'r', encoding='utf-8') as file:
    vocabulary = [line.strip() for line in file.readlines()]

# Initialize TweetCleaner and TweetFeatureExtractor
tweet_cleaner = TweetCleaner()
feature_extractor = TweetFeatureExtractor(vocabulary)

# Take tweet as input argument
if len(sys.argv) < 2:
    print("Usage: python predicter.py <tweet>")
    sys.exit(1)

tweet = sys.argv[1]

# Clean the tweet
cleaned_tweet = tweet_cleaner.clean_tweet(tweet)
print("\nCleaned tweet:", cleaned_tweet)

# Generate feature vector for the tweet
feature_vector = feature_extractor.binary_feature_representation_for_tweet(cleaned_tweet)
print("\nFeature vector for the tweet:", feature_vector)

# Predict
pred = model.predict(np.array([feature_vector, ]))
print("pred",float(pred[0]))
if(pred[0] > 0.5) :
    print("\n>>> this is a positive tweet!")
else :
    print("\n>>> this is a negative tweet!")

