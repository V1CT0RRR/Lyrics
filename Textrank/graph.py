import math
from collections import defaultdict

import matplotlib.pyplot as plt
import networkx as nx


class TextrankGraph:

    def __init__(self, preprocessed, desired_tags, k=10, window_size=2, modified=True):        
        """ Initialize a TextrankGraph object

        Args:
            preprocessed (Preprocessed): A Preprocessed object storing info of the document.
            desired_tags (List[str]): Desired tags to keep in the graph..
            k (int, optional): Number of key phrase candidates. Defaults to 10.
            modified (bool, optional): Use modified post-processing. Defaults to True.
        """

        self.k = k
        self.window_size = window_size
        self.modified = modified

        self.desired_tags = desired_tags

        self.graph = nx.Graph()
        self.seen_lemma = defaultdict(int)

        self.preprocessed = preprocessed

        self.pipeline()


    def process_sentence(self, sentence):
        """ Process a sentence to initialize the graph.

        Args:
            sentence (List[(str, str, str)]): A sentence in the form of (token, tag, lemma) pairs.
        """

        for idx, (_, tag, lemma) in enumerate(sentence):
            if tag not in self.desired_tags:
                continue

            key = (lemma, tag)

            self.seen_lemma[key] += 1

            node_id = list(self.seen_lemma.keys()).index(key)

            if not node_id in self.graph.nodes:
                self.graph.add_node(node_id)

            for prev_idx in range(idx-1, max(-1, idx-self.window_size), -1):
                _, prev_tag, prev_lemma = sentence[prev_idx]
                prev_key = (prev_lemma, prev_tag)
                if prev_key not in self.seen_lemma or prev_key == key:
                    continue
                prev_node_id = list(self.seen_lemma.keys()).index(prev_key)
                if self.graph.has_edge(node_id, prev_node_id):
                    self.graph[node_id][prev_node_id]["weight"] += 1
                else:
                    self.graph.add_edge(node_id, prev_node_id, weight=1)


    def retrieve_phrases(self, textrank, T, merge_window_size=2):
        """ Retrieve scores for phrases from single-word textranks.

        Args:
            textrank (List[(str, float)]): List of single words with their ranks.
            T (int): Number of candidate single words to keep for merging.
            merge_window_size (int, optional): Window size for merging. Defaults to 2.

        Returns:
            List(str, float): List of (key phrase, score) pairs as the final result
        """

        candidates = sorted(textrank.items(), key=lambda x: x[1], reverse=True)[:T]

        candidate_keys = {list(self.seen_lemma.keys())[candidate[0]]: candidate[1] for candidate in candidates}

        merged_level = 1
        max_merged_level = 3
        while True:
            found_new = False
            for sentence in self.preprocessed.sentences:
                sentence_length = len(sentence)
                for idx in range(sentence_length):
                    if idx + merged_level + merge_window_size - 1 > sentence_length:
                        break

                    merge_window = sentence[idx:idx+merged_level+merge_window_size-1]
                    merge_window_lemma = merge_window_test = [(lemma, tag) for (word, tag, lemma) in merge_window]

                    if merged_level > 1:
                        merge_window_test = [tuple(merge_window_test[0:merged_level])] + merge_window_test[merged_level:]

                    if all(key in candidate_keys.keys() for key in merge_window_test):
                        found_new = True
                        candidate_keys[tuple(merge_window_lemma)] = sum([candidate_keys[key] for key in merge_window_test])

            if not found_new or merged_level >= max_merged_level:
                break
            merged_level += 1

        candidate_keys = sorted(candidate_keys.items(), key=lambda x: x[1], reverse=True)[:self.k]

        candidate_keys = [(' '.join([word for (word, _) in key]) if type(key[0]) != str else key[0], rank) for key, rank in candidate_keys]

        return candidate_keys


    def add_phrase_rank(self, phrase, textrank, candidates):
        """ Add compound noun phrases and compute ranks

        Args:
            phrase (List(tuple(str, str, str))): List of (token, tag, lemma) tuples that belongs to a phrase.
            textrank (dict): Dictionary of scores of single words.
            candidates (dict): Single keywords or compound key phrases and their scores to be updated

        Returns:
            dict: Updated single keywords or compound key phrases and their scores
        """

        phrase_len = len(phrase)

        if phrase_len == 0:
            return candidates

        rank = 0
        non_lemma = 0
        merged_key = []

        for (_, tag, lemma) in phrase:
            key = (lemma, tag)
            if key in self.seen_lemma.keys():
                node_id = list(self.seen_lemma.keys()).index(key)
                rank += textrank[node_id]
                merged_key.append(key)
            else:
                non_lemma += 1

        if len(merged_key) <= 0:
            return candidates

        # https://github.com/DerwenAI/pytextrank/blob/7cc079b1856e59cc3e4b53268a01b5e8893ca1ae/pytextrank/base.py#L582
        discount = phrase_len / (phrase_len + (2.0 * non_lemma) + 1.0)
        rank = math.sqrt(rank / (phrase_len + non_lemma)) * discount

        merged_key = tuple((merged_key)) if len(merged_key) > 1 else tuple(merged_key[0])

        self.seen_lemma[merged_key] += 1

        # if merged_key not in candidates.keys():
        #     candidates[merged_key] = rank

        phrase_text = " ".join([word for (word, _, _) in phrase]).lower().replace("'", "")
        if merged_key not in candidates.keys():
            candidates[merged_key] = set([(phrase_text, rank)])
        else:
            candidates[merged_key].add((phrase_text, rank))

        return candidates


    def retrieve_phrases_from_np(self, textrank):
        """ Compute scores for compound key phrases based on textrank

        Args:
            textrank (dict): Dictionary of scores of single words

        Returns:
            List(tuple(str, float)): List of (key phrase, score) pairs
        """

        candidates = defaultdict(float)
        for sentence in self.preprocessed.np_chunks:
            for np_chunk in sentence:
                candidates = self.add_phrase_rank(np_chunk, textrank, candidates)

        results = []
        for key, rank_set in candidates.items():
            max_phrase, max_rank = max(rank_set, key=lambda x: x[1])
            results.append((max_phrase, max_rank))

        return sorted(results, key=lambda x: x[1], reverse=True)[:self.k]

        # candidates = sorted(candidates.items(), key=lambda x: x[1], reverse=True)[:self.k]

        # candidates = [(' '.join([word for (word, _) in key]) if type(key[0]) != str else key[0], rank) for key, rank in candidates]

        # return candidates

    
    def draw_graph(self):
        """ Draw constructed graph
        """

        plt.figure(figsize=(10, 10))

        labels = {idx: key[0] for idx, key in enumerate(list(self.seen_lemma.keys())) if idx in self.graph.nodes and type(key[0]) is str}

        pos = nx.spring_layout(self.graph)

        nx.draw(self.graph, pos=pos)
        nx.draw_networkx_labels(self.graph, pos, labels)

        plt.savefig("../imgs/textrank.png")
        plt.show()

    
    def print_textrank(self):
        """ Print textrank of single words
        """

        print()
        for key, val in sorted(self.textrank.items(), key=lambda x: x[1], reverse=True):
            lemma, _ = list(self.seen_lemma.keys())[key]
            print("{:<4f}: {}".format(val, lemma))


    def pipeline(self):
        """ Pipeline for graph constructing and rank computing
        """

        for sentence in self.preprocessed.sentences:
            self.process_sentence(sentence)

        self.textrank = nx.pagerank(self.graph)

        if self.modified:
            self.candidates = self.retrieve_phrases_from_np(self.textrank)
        else:
            self.candidates = self.retrieve_phrases(self.textrank, T=self.graph.number_of_nodes())