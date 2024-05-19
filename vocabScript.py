import csv

class VocabularyGenerator:
    def __init__(self, cleaned_data, k):
        self.cleaned_data = cleaned_data
        self.k = k
        self.vocabulary = self.generer_vocab()

    def generer_vocab(self):
        word_count = {}
        with open(self.cleaned_data, 'r', encoding='utf-8') as csv_file:
            reader = csv.reader(csv_file)
            for row in reader:
                tweet = row[5]
                for word in tweet.split():
                    word_count[word] = word_count.get(word, 0) + 1
        
        vocabulary = [word for word, count in word_count.items() if count >= self.k]
        return vocabulary

    def map_words_to_indices(self):
        word_to_index = {word: index for index, word in enumerate(self.vocabulary)}
        indices = []
        with open(self.cleaned_data, 'r', encoding='utf-8') as csv_file:
            reader = csv.reader(csv_file)
            for row in reader:
                tweet = row[5]
                tweet_indices_row = []
                for word in tweet.split():
                    if word in word_to_index:
                        tweet_indices_row.append(word_to_index[word])
                indices.append(tweet_indices_row)        
        return indices

    def save_vocabulary(self, vocab_file):
        with open(vocab_file, 'w', encoding='utf-8') as file:
            for word in self.vocabulary:
                file.write(word + '\n')

    def write_indices_to_file(self, tweet_indices, output_file):
        with open(output_file, 'w', encoding='utf-8') as file:
            for indices_row in tweet_indices:
                file.write(','.join(map(str, indices_row)) + '\n')

