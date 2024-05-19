import gc
import sys

import numpy as np
import pandas as pd
from features import TweetFeatureExtractor
import tensorflow as tf
from tweet_clean import TweetCleaner

class Deep_Learning_Classifier :
    def __init__(self, df) :
        self.df = df

    # load the model from existing save
    def loadModel(self, path) :
        self.model = tf.keras.models.load_model(path)

    def split_df(self, df : pd):

        print("\nsplitting data ...")

        train_df = df.sample(frac=0.8, random_state=42)
        print(train_df.head())

        test_df = df.drop(train_df.index)

        self.X_train = train_df.iloc[:, 1:]
        self.y_train = train_df.iloc[:, 0]
        self.X_test = test_df.iloc[:, 1:]
        self.y_test = test_df.iloc[:, 0]

        del df
        del test_df
        del train_df

        gc.collect()

    # generate the model from dataframe
    def generateModel(self) :

        # setting the nn model
        print("\nsetting model ...")
        input_shape = [self.X_train.shape[1]]
        model = tf.keras.Sequential([

            ## setup the model layers
            ### input layer with 64 nodes and relu activation
            tf.keras.layers.Dense(64, activation='relu', input_shape = (X_train.shape[1],)),
            ### hidden layer with 16 nodes and relu activation
            tf.keras.layers.Dense(16, activation='relu'),
            ### input layer with 1 node and sigmoid activation
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        print("\ncompiling model ...")
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

        # loading the model / weights
        # print("\nloading model ...")
        # model = tf.keras.models.load_model('models/DLClassifier-v001')
        # model.load_weights('path/to/dl_weights.h5')

        # summery
        model.summary()

    
    def predict_one(self, text) :
        
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
        print("Cleaned tweet:", cleaned_tweet)

        # Generate feature vector for the tweet
        feature_vector = feature_extractor.binary_feature_representation_for_tweet(cleaned_tweet)
        print("Feature vector for the tweet:", feature_vector)

        try :
            pred_proba = self.model.predict(feature_vector)
            pred = np.array([1 if x >= 0.5 else 0 for x in pred_proba])
        except :
            print("\ntry Loading the tensorflow model first!")

        if(pred[0] == 1) :
            print("this is a positive tweet.")
        else :
            print("this is a negative tweet.")

    # predict
    def predict_many_proba(self, X_test) :

        y_pred_proba = self.model.predict(X_test)

        return y_pred_proba
    

