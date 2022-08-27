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
                 doc_seg=True,
                 emb_align=True,
                 nlp=None,
                 data_path_word_weight = os.path.join(dirname, "../models/enwiki_vocab_min200.txt"),
                 data_path_word_weight_inspec = os.path.join(dirname, "../models/inspec_vocab.txt"),
                 data_path_elmo_weights = os.path.join(dirname, "../models/elmo/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"),
                 data_path_elmo_options = os.path.join(dirname, "../models/elmo/elmo_2x4096_512_2048cnn_2xhighway_options.json")):
        """ Initialzie a SIFRank object.

        Args:
            k (int, optional): Number of keywords to extract. Defaults to 10.
            doc_seg (bool, optional): Use document segmentation. Defaults to True.
            emb_align (bool, optional): Use embedding alignment. Defaults to False.
            nlp (_type_, optional): Spacy language model for preprocessing. Defaults to None.
            data_path_word_weight (str, optional): path to word weight file. Defaults to os.path.join(dirname, "../models/enwiki_vocab_min200.txt").
            data_path_word_weight_inspec (str, optional): path to word weight inspec file. Defaults to os.path.join(dirname, "../models/inspec_vocab.txt").
            data_path_elmo_weights (str, optional): path to elmo weights file. Defaults to os.path.join(dirname, "../models/elmo/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5").
            data_path_elmo_options (str, optional): path to elmo options file. Defaults to os.path.join(dirname, "../models/elmo/elmo_2x4096_512_2048cnn_2xhighway_options.json").
        """

        self.considered_tags = DISIRED_TAGS_3_NLTK
        self.lemmatizer = nltk.WordNetLemmatizer().lemmatize

        self.k = k

        self.init_weights(data_path_word_weight, data_path_word_weight_inspec)
        self.init_embeddor(data_path_elmo_weights, data_path_elmo_options)

        self.doc_seg = doc_seg
        self.emb_align = emb_align

        self.nlp = nlp


    def get_init_word_weights(self, file="", weight_param=2.7e-4):
        """ Retrieve word weights from corpus file.

        Args:
            file (str, optional): path to word weight file. Defaults to "".
            weight_param (float, optional): parameter controlling weight computation. Defaults to 2.7e-4.

        Returns:
            dict: matches a word to its weight.
        """        

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
        """ Initialize word weights.

        Args:
            data_path_word_weight (str): path to word weight file.
            data_path_word_weight_inspec (str): path to word weight inspec file.
        """        

        self.word_weight_pretrained = self.get_init_word_weights(data_path_word_weight)
        self.word_weight_finetuned = self.get_init_word_weights(data_path_word_weight_inspec)


    def init_embeddor(self, data_path_elmo_weights, data_path_elmo_options, embeddor_type='elmo'):
        """ Initialize embeddor for encoding.

        Args:
            data_path_elmo_weights (str): path to elmo weights file.
            data_path_elmo_options (str): path to elmo options file.
            embeddor_type (str, optional): type of embeddor to use. Defaults to 'elmo'.
        """

        if embeddor_type == 'elmo':
            self.embeddor = ElmoEmbedder(data_path_elmo_options, data_path_elmo_weights)


    def prepare_sent_segments(self, tokenized_sent):
        """ Segment sentences to speed up embedding.

        Args:
            tokenized_sent (List(List(str))): List of sentences of tokens.

        Returns:
            List(List(str)): List of batches of tokens.
        """        

        min_seq_len = 16

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
        """ Compute sentence embedding using initialized embeddor.

        Args:
            tokenized_sent (List(List(str))): List of batches of tokens.

        Returns:
            torch.Tensor: embedding in shape (batch_size, 3, longest_sentence_length, 1024).
        """

        elmo_embeddings, elmo_mask = self.embeddor.batch_to_embeddings(tokenized_sent)
        return elmo_embeddings

    
    def embedding_alignment(self, elmo_embeddings, sent_segments):
        """ Compute aligned embedding.

        Args:
            elmo_embeddings (torch.Tensor): embedding in shape (batch_size, 3, longest_sentence_length, 1024).
            sent_segments (List(List(str))): List of batches of tokens.
        """        
        
        token2embedding = {}
        for i, batch in enumerate(sent_segments):
            for j, token in enumerate(batch):
                embedding = elmo_embeddings[i, 1, j, :]
                if token not in token2embedding:
                    token2embedding[token] = [embedding]
                else:
                    token2embedding[token].append(embedding)

        for token in token2embedding.keys():
            token2embedding[token] = torch.stack(token2embedding[token]).sum(dim=0) / float(len(token2embedding[token]))

        for i in range(0, elmo_embeddings.shape[0]):
            for j, token in enumerate(sent_segments[i]):
                embedding = token2embedding[token]
                elmo_embeddings[i, 2, j :] = embedding

        return elmo_embeddings


    def merge_seg_sent_embedding(self, elmo_embeddings, sent_segments):
        """ Merge embeddings of segmented sentences.

        Args:
            elmo_embeddings (torch.Tensor): embedding in shape (batch_size, 3, longest_sentence_length, 1024).
            sent_segments (List(List(str))): List of batches of tokens.

        Returns:
            torch.Tensor: embedding in shape (1, 3, num_tokens, 1024).
        """

        new_embeddings = [elmo_embeddings[i:i+1, :, 0:len(sent_segments[i]), :] for i in range(len(sent_segments))]
        new_embeddings = torch.cat(new_embeddings, dim=2)
        return new_embeddings


    def get_tokenized_sent_weight(self, tokenized_sent, word_weight):
        """ Get weights for each word in a sentence.

        Args:
            tokenized_sent (List(List(str))): List of batches of tokens.
            word_weight (dict): words and their weights.

        Returns:
            List(float): List of corresponding weights of words in the sentence.
        """

        weights = []

        for word in tokenized_sent:
            word = self.lemmatizer(word.lower())
            weight = 0.0
            if word in word_weight:
                weight = word_weight[word]
            weights.append(weight)

        return weights


    def get_tokenized_sent_weight_average(self, tokens, tagged_tokens, word_weight, embedding_list):
        """ Compute weighted average of tokenized sentence.

        Args:
            tokens (List(str)): List of tokens.
            tagged_tokens (List(tuple(str, str))): List of tagged tokens.
            word_weight (dict): words and their weights.
            embedding_list (torch.Tensor): embedding tensor.

        Returns:
            torch.Tensor: weighted average sentence embedding.
        """

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
        """ Compute weighted average of candidate.

        Args:
            word_weight (dict): words and their weights.
            embedding_list (torch.Tensor): embedding tensor.
            start (int): start index of the candidate phrase in the document.
            end (int): end index of the candidate phrase in the document.

        Returns:
            torch.Tensor: weighted average candidate embedding.
        """        

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
        """ Get sentence embedding and candidate embeddings.

        Args:
            preprocessed (Preprocessing): preprocessed document.

        Returns:
            tuple(torch.Tensor, List(torch.Tensor)): sentence embedding and candidate embeddings.
        """

        if not self.doc_seg:
            elmo_embeddings = self.get_sent_embedding([preprocessed.tokens])
        elif not self.emb_align:
            sent_segments = self.prepare_sent_segments(preprocessed.tokenized_sentences)
            elmo_embeddings = self.get_sent_embedding(sent_segments)
            elmo_embeddings = self.merge_seg_sent_embedding(elmo_embeddings, sent_segments)
        else:
            sent_segments = self.prepare_sent_segments(preprocessed.tokenized_sentences)
            elmo_embeddings = self.get_sent_embedding(sent_segments)
            elmo_embeddings = self.embedding_alignment(elmo_embeddings, sent_segments)
            elmo_embeddings = self.merge_seg_sent_embedding(elmo_embeddings, sent_segments)

        word_weight = self.get_tokenized_sent_weight(preprocessed.tokens, self.word_weight_pretrained)
        sent_embedding = self.get_tokenized_sent_weight_average(preprocessed.tokens, preprocessed.tagged_tokens, word_weight, elmo_embeddings[0])

        candidate_embedding_list = []
        for candidate, (start, end) in preprocessed.np_candidates:
            candidate_embedding = self.get_candidate_weight_average(word_weight, elmo_embeddings[0], start, end)
            candidate_embedding_list.append(candidate_embedding)

        return sent_embedding, candidate_embedding_list


    def get_ranks(self, preprocessed, sent_embedding, candidate_embedding_list):
        """ Compute ranks based on sentence embedding and candidate embeddings.

        Args:
            preprocessed (Preprocessing): preprocessed document.
            sent_embedding (torch.Tensor): weighted average sentence embedding.
            candidate_embedding_list (List(torch.Tensor)): weighted average candidate embeddings.

        Returns:
            List(tuple(str, float)): candidates and their scores
        """        

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
        """ Pipeline for the SIFRank algorithm.

        Args:
            document (str): Document to find keywords from.
        """

        preprocessed = Preprocessing(document, nlp=self.nlp)

        sent_embedding, candidate_embedding_list = self.get_sent_candidate_embedding(preprocessed)

        self.result = self.get_ranks(preprocessed, sent_embedding, candidate_embedding_list)

        for word, rank in self.result:
            print("{:.5f}: {}".format(rank, word))


if __name__ == '__main__':
    text = "Discrete output feedback sliding mode control of second order systems - a moving switching line approach The sliding mode control systems (SMCS) for which the switching variable is designed independent of the initial conditions are known to be sensitive to parameter variations and extraneous disturbances during the reaching phase. For second order systems this drawback is eliminated by using the moving switching line technique where the switching line is initially designed to pass the initial conditions and is subsequently moved towards a predetermined switching line. In this paper, we make use of the above idea of moving switching line together with the reaching law approach to design a discrete output feedback sliding mode control. The main contributions of this work are such that we do not require to use system states as it makes use of only the output samples for designing the controller. and by using the moving switching line a low sensitivity system is obtained through shortening the reaching phase. Simulation results show that the fast output sampling feedback guarantees sliding motion similar to that obtained using state feedback"

    SIFRank(doc_seg=True).pipeline(document=text)