import sys
from projetRN.tweet_clean import TweetCleaner
from features import TweetFeatureExtractor

# Read vocabulary from file
with open('vocab.txt', 'r', encoding='utf-8') as file:
    vocabulary = [line.strip() for line in file.readlines()]

# Initialize TweetCleaner and TweetFeatureExtractor
tweet_cleaner = TweetCleaner()
feature_extractor = TweetFeatureExtractor(vocabulary)

# Take tweet as input argument
if len(sys.argv) < 2:
    print("Usage: python script.py <tweet>")
    sys.exit(1)

tweet = sys.argv[1]

# Clean the tweet
cleaned_tweet = tweet_cleaner.clean_tweet(tweet)
print("Cleaned tweet:", cleaned_tweet)

# Generate feature vector for the tweet
feature_vector = feature_extractor.binary_feature_representation_for_tweet(cleaned_tweet)
print("Feature vector for the tweet:", feature_vector)
