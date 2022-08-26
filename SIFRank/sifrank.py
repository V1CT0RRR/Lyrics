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
    def __init__(self, k=10,
                 data_path_word_weight = os.path.join(dirname, "../models/enwiki_vocab_min200.txt"),
                 data_path_word_weight_inspec = os.path.join(dirname, "../models/inspec_vocab.txt"),
                 data_path_elmo_weights = os.path.join(dirname, "../models/elmo/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"),
                 data_path_elmo_options = os.path.join(dirname, "../models/elmo/elmo_2x4096_512_2048cnn_2xhighway_options.json"),
                 doc_seg=True,
                 emb_align=False):

        self.considered_tags = DISIRED_TAGS_3_NLTK
        self.lemmatizer = nltk.WordNetLemmatizer().lemmatize

        self.k = k

        self.init_weights(data_path_word_weight, data_path_word_weight_inspec)
        self.init_embeddor(data_path_elmo_weights, data_path_elmo_options)

        self.doc_seg = doc_seg
        self.emb_align = emb_align


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


    def prepare_sent_segments(self, tokenized_sent):
        min_seq_len = 16
        sent_segments = []

        batch = []
        instance = []

        for sentence in tokenized_sent:
            if len(instance) >= min_seq_len:
                batch.append(instance)
                instance = sentence
            else:
                instance += sentence
        batch.append(instance)

        return batch


    def get_sent_embedding(self, tokenized_sent):
        elmo_embeddings, elmo_mask = self.embeddor.batch_to_embeddings(tokenized_sent)
        return elmo_embeddings


    def merge_seg_sent_embedding(self, elmo_embeddings, sent_segments):
        new_embeddings = [elmo_embeddings[i:i+1, :, 0:len(sent_segments[i]), :] for i in range(len(sent_segments))]
        new_embeddings = torch.cat(new_embeddings, dim=2)
        return new_embeddings


    def get_tokenized_sent_weight(self, tokenized_sent, word_weight):
        weights = []

        for word in tokenized_sent:
            word = self.lemmatizer(word.lower())
            weight = 0.0
            if word in word_weight:
                weight = word_weight[word]
            weights.append(weight)

        return weights


    def get_tokenized_sent_weight_average(self, tokens, tagged_tokens, word_weight, embedding_list):
        num_words = len(tokens)

        e_test_list = []
        result = torch.zeros((3, 1024))
        for i in range(0, 3):
            for j in range(num_words):
                if tagged_tokens[j][1] in self.considered_tags:
                    e_test = embedding_list[i][j]
                    e_test_list.append(e_test)
                    result[i] += e_test * word_weight[j]
            result[i] = result[i] / float(num_words)
        return result


    def get_candidate_weight_average(self, word_weight, embedding_list, start, end):
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

        if not self.doc_seg:
            elmo_embeddings = self.get_sent_embedding([preprocessed.tokens])
        else:
            sent_segments = self.prepare_sent_segments(preprocessed.tokenized_sentences)
            elmo_embeddings = self.get_sent_embedding(sent_segments)
            elmo_embeddings = self.merge_seg_sent_embedding(elmo_embeddings, sent_segments)

        word_weight = self.get_tokenized_sent_weight(preprocessed.tokens, self.word_weight_pretrained)
        sent_embedding = self.get_tokenized_sent_weight_average(preprocessed.tokens, preprocessed.tagged_tokens, word_weight, elmo_embeddings[0])

        candidate_embedding_list = []
        for candidate, (start, end) in preprocessed.np_candidates:
            candidate_embedding = self.get_candidate_weight_average(word_weight, elmo_embeddings[0], start, end)
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

        self.result = self.get_ranks(preprocessed, sent_embedding, candidate_embedding_list)

        for word, rank in self.result:
            print("{:.5f}: {}".format(rank, word))


if __name__ == '__main__':
    text = "Discrete output feedback sliding mode control of second order systems - a moving switching line approach The sliding mode control systems (SMCS) for which the switching variable is designed independent of the initial conditions are known to be sensitive to parameter variations and extraneous disturbances during the reaching phase. For second order systems this drawback is eliminated by using the moving switching line technique where the switching line is initially designed to pass the initial conditions and is subsequently moved towards a predetermined switching line. In this paper, we make use of the above idea of moving switching line together with the reaching law approach to design a discrete output feedback sliding mode control. The main contributions of this work are such that we do not require to use system states as it makes use of only the output samples for designing the controller. and by using the moving switching line a low sensitivity system is obtained through shortening the reaching phase. Simulation results show that the fast output sampling feedback guarantees sliding motion similar to that obtained using state feedback"
    
    text = '''
    Does social capital determine innovation? To what extent?
This paper deals with two questions: Does social capital determine innovation
	in manufacturing firms? If it is the case, to what extent? To deal with
	these questions, we review the literature on innovation in order to see
	how social capital came to be added to the other forms of capital as an
	explanatory variable of innovation. In doing so, we have been led to
	follow the dominating view of the literature on social capital and
	innovation which claims that social capital cannot be captured through
	a single indicator, but that it actually takes many different forms
	that must be accounted for. Therefore, to the traditional explanatory
	variables of innovation, we have added five forms of structural social
	capital (business network assets, information network assets, research
	network assets, participation assets, and relational assets) and one
	form of cognitive social capital (reciprocal trust). In a context where
	empirical investigations regarding the relations between social capital
	and innovation are still scanty, this paper makes contributions to the
	advancement of knowledge in providing new evidence regarding the impact
	and the extent of social capital on innovation at the two
	decisionmaking stages considered in this study
    '''

    SIFRank(doc_seg=True).pipeline(document=text)