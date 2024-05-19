import csv




class TweetFeatureExtractor:
    def __init__(self, vocabulary):
        self.vocabulary = vocabulary

    def binary_feature_representation_for_tweet(self, tweet):
        feature_vector = [1 if word in tweet else 0 for word in self.vocabulary]
        return feature_vector

    def binary_feature_representation_for_data(self, data_file, output_file, indices_file):
        with open(output_file, 'w', newline='', encoding='utf-8') as output_csv:
            writer = csv.writer(output_csv)
            with open(data_file, 'r', encoding='utf-8') as data_csv, open(indices_file, 'r', encoding='utf-8') as indices_csv:
                data_reader = csv.reader(data_csv)
                indices_reader = csv.reader(indices_csv)
                for data_row, indices_row in zip(data_reader, indices_reader):
                    feature_vector = [0 if data_row[0] == '0' else 1]
                    feature_vector += [1 if str(index) in indices_row else 0 for index in range(len(self.vocabulary))]
                    writer.writerow(feature_vector)

# Exemple d'utilisation :
# Supposons que "data" est une liste contenant tous vos tweets prétraités
# et "vocabulary" est votre liste de mots de vocabulaire



#cleaned_data ='cleaned_data.csv'
#indices='indices.txt'
#with open('vocab.txt', 'r', encoding='utf-8') as file:
 #       vocabulary = [line.strip() for line in file.readlines()]
#print(vocabulary)
#ex=TweetFeatureExtractor(vocabulary)
#binary_feature_vectors = ex.binary_feature_representation_for_data(cleaned_data,"final_data.csv",indices)
