from pathlib import Path
import os

import nltk
import torch

from allennlp.commands.elmo import ElmoEmbedder

from preprocessing import Preprocessing
import utils

# Desired tags for SIFRank

# nltk tags
DISIRED_TAGS_1_NLTK =           ["JJ", "NN", "NNS", "NNP", "NNPS"]
DISIRED_TAGS_2_NLTK =           ["JJ", "NN", "NNS", "NNP", "NNPS", "VB", "VBG", "VBD", "VBN", "VBP", "VBZ"]
DISIRED_TAGS_3_NLTK =           ["JJ", "NN", "NNS", "NNP", "NNPS", "VBG"]

# universal tags
DISIRED_TAGS_1_UNIVERSAL =      ["ADJ", "NOUN", "PROPN"]
DISIRED_TAGS_2_UNIVERSAL =      ["ADJ", "NOUN", "PROPN", "VERB"]

dirname = os.path.abspath(os.path.dirname(__file__))

class SIFRank:
    def __init__(self, document, k=10,
                 data_path_word_weight = os.path.join(dirname, '../data/enwiki_vocab_min200.txt'),
                 data_path_word_weight_inspec = os.path.join(dirname, '../data/inspec_vocab.txt'),
                 data_path_elmo_weights = os.path.join(dirname, "../models/elmo/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"),
                 data_path_elmo_options = os.path.join(dirname, "../models/elmo/elmo_2x4096_512_2048cnn_2xhighway_options.json")):

        self.considered_tags = DISIRED_TAGS_3_NLTK
        self.lemmatizer = nltk.WordNetLemmatizer().lemmatize

        self.k = k

        self.init_weights(data_path_word_weight, data_path_word_weight_inspec)
        self.init_embeddor(data_path_elmo_weights, data_path_elmo_options)

        self.pipeline(document)

    def get_init_word_weights(self, file="", weight_param=2.7e-4):
        if weight_param <= 0:
            weight_param = 1

        lines = []
        with open(file) as f:
            lines = f.readlines()

        word_weight = dict()
        word_count = dict()
        word_count_total = 0

        for line in lines:
            word_and_count = line.strip().split()
            if len(word_and_count) != 2:
                continue
            word, count = word_and_count
            word_count[word] = float(count)
            word_count_total += float(count)

        for word, count in word_count.items():
            word_weight[word] = weight_param / (weight_param + count / word_count_total)

        return word_weight

    def init_weights(self, data_path_word_weight, data_path_word_weight_inspec):
        self.word_weight_pretrained = self.get_init_word_weights(data_path_word_weight)
        self.word_weight_finetuned = self.get_init_word_weights(data_path_word_weight_inspec)

    def init_embeddor(self, data_path_elmo_weights, data_path_elmo_options, embeddor_type='elmo'):
        if embeddor_type == 'elmo':
            self.embeddor = ElmoEmbedder(data_path_elmo_options, data_path_elmo_weights)

    def get_sent_embedding(self, tokenized_sent):
        elmo_embeddings, elmo_mask = self.embeddor.batch_to_embeddings([tokenized_sent])
        return elmo_embeddings

    def get_tokenized_sent_weight(self, tokenized_sent, word_weight):
        weights = []

        for word in tokenized_sent:
            word = self.lemmatizer(word.lower())
            weight = 0.0
            if word in word_weight:
                weight = word_weight[word]
            weights.append(weight)

        return weights

    def get_tokenized_sent_weight_average(self, tokenized_sent, tokenized_tagged_sent, word_weight, embedding_list):
        num_words = len(tokenized_sent)

        e_test_list = []
        result = torch.zeros((3, 1024))
        for i in range(0, 3):
            for j in range(num_words):
                if tokenized_tagged_sent[j][1] in self.considered_tags:
                    e_test = embedding_list[i][j]
                    e_test_list.append(e_test)
                    result[i] += e_test * word_weight[j]
                
            result[i] = result[i] / float(num_words)
        return result

    def get_candidate_weight_average(self, tokenized_sent, word_weight, embedding_list, start, end):
        num_words = end - start

        e_test_list = []
        result = torch.zeros((3, 1024))
        for i in range(0, 3):
            for j in range(start, end):
                e_test = embedding_list[i][j]
                e_test_list.append(e_test)
                result[i] += e_test * word_weight[j]
            result[i] = result[i] / float(num_words)
        return result

    def get_sent_candidate_embedding(self, preprocessed):
        elmo_embeddings = self.get_sent_embedding(preprocessed.tokenized_sentences)

        word_weight = self.get_tokenized_sent_weight(preprocessed.tokenized_sentences, self.word_weight_pretrained)
        sent_embedding = self.get_tokenized_sent_weight_average(preprocessed.tokenized_sentences, preprocessed.tokenized_tagged_sentences, word_weight, elmo_embeddings[0])

        candidate_embedding_list = []
        for candidate, (start, end) in preprocessed.np_candidates:
            candidate_embedding = self.get_candidate_weight_average(preprocessed.tokenized_sentences, word_weight, elmo_embeddings[0], start, end)
            candidate_embedding_list.append(candidate_embedding)

        return sent_embedding, candidate_embedding_list

    def get_ranks(self, preprocessed, sent_embedding, candidate_embedding_list):
        candidate_distances = []
        for idx, candidate_embedding in enumerate(candidate_embedding_list):
            distance = utils.get_distance_cosine(sent_embedding, candidate_embedding)
            candidate_distances.append(distance)

        dict_candidate_distances = {}
        for idx in range(len(candidate_distances)):
            candidate, _ = preprocessed.np_candidates[idx]
            candidate_norm = ' '.join([nltk.WordNetLemmatizer().lemmatize(word) for word in candidate.split()])
            if candidate in dict_candidate_distances:
                dict_candidate_distances[candidate_norm].append(candidate_distances[idx])
            else:
                dict_candidate_distances[candidate_norm] = [candidate_distances[idx]]

        for key, value in dict_candidate_distances.items():
            dict_candidate_distances[key] = sum(value) / len(value)

        return sorted(dict_candidate_distances.items(), key=lambda x: x[1], reverse=True)[:20]

    def pipeline(self, document):
        preprocessed = Preprocessing(document)

        sent_embedding, candidate_embedding_list = self.get_sent_candidate_embedding(preprocessed)

        result = self.get_ranks(preprocessed, sent_embedding, candidate_embedding_list)

        for word, rank in result:
            print("{:.5f}: {}".format(rank, word))


if __name__ == '__main__':
    text = "Discrete output feedback sliding mode control of second order systems - a moving switching line approach The sliding mode control systems (SMCS) for which the switching variable is designed independent of the initial conditions are known to be sensitive to parameter variations and extraneous disturbances during the reaching phase. For second order systems this drawback is eliminated by using the moving switching line technique where the switching line is initially designed to pass the initial conditions and is subsequently moved towards a predetermined switching line. In this paper, we make use of the above idea of moving switching line together with the reaching law approach to design a discrete output feedback sliding mode control. The main contributions of this work are such that we do not require to use system states as it makes use of only the output samples for designing the controller. and by using the moving switching line a low sensitivity system is obtained through shortening the reaching phase. Simulation results show that the fast output sampling feedback guarantees sliding motion similar to that obtained using state feedback"
    SIFRank(document=text)