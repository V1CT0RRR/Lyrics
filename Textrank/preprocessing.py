import re
import nltk


# Grammars for noun phrase parsing 

# nltk tags
GRAMMAR_1_NLTK =        "NP: {<DT>?<NN.*|JJ.*>*<NN.*>}"
GRAMMAR_2_NLTK =        "NP: {<DT>?<JJ.*>*<NN*>}"
GRAMMAR_3_NLTK =        "NP: {<NN.*|JJ.*>*<NN.*>}"

# universal tags
GRAMMAR_1_UNIVERSAL =   "NP: {<DET>?<NOUN|ADJ>*<NOUN*>}"
GRAMMAR_2_UNIVERSAL =   "NP: {<DET>?<ADJ>*<NOUN*>}"
GRAMMAR_3_UNIVERSAL =   "NP: {<NOUN|ADJ>*<NOUN*>}"


class Preprocessing:
    def __init__(self, document,
                 tokenizer=nltk.tokenize.word_tokenize,
                 tagger=nltk.tag.pos_tag,
                 lemmatizer = nltk.stem.WordNetLemmatizer().lemmatize,
                 np_grammar=GRAMMAR_1_NLTK):
        """ Initialize a Preprocessing object

        Args:
            document (str): Document to preprocess.
            tokenizer (function, optional): NLTK-style tokenizer. Defaults to nltk.tokenize.word_tokenize.
            tagger (function, optional): NLTK-style tagger. Defaults to nltk.tag.pos_tag.
            lemmatizer (function, optional): NLTK-style lemmatizer. Defaults to nltk.stem.WordNetLemmatizer().lemmatize.
            np_grammar (str, optional): Grammar for noun parsing. Defaults to GRAMMAR_1_NLTK.
        """

        self.stopwords = set(nltk.corpus.stopwords.words("english"))

        self.tokenizer = tokenizer
        self.tagger = tagger
        self.lemmatizer = lemmatizer

        self.np_grammar = np_grammar
        self.np_grammar_parser = nltk.RegexpParser(self.np_grammar)

        self.pipeline(document)

    def clean(self, document):
        """ Clean a document by splitting sentences and replacing invalid characters.

        Args:
            document (str): Document string to clean.

        Returns:
            List[str]: List of cleaned sentences.
        """

        sentences = document.replace('\n', '. ').split('. ')
        for idx, sentence in enumerate(sentences):
            sentences[idx] = re.sub("[^a-zA-Z0-9'.,\- ]", '', sentence)#.lower()
        return sentences

    def tokenize(self, sentences, remove_stopwords=False):
        """ Tokenize sentences with the initialized tokenizer.

        Args:
            sentences (List[str]): List of sentences to be tokenized.
            remove_stopwords (bool, optional): Remove stopwords. Defaults to False.

        Returns:
            List[List[str]]: List of sentences in the form of list of tokens.
        """

        tokenized_sentences = []
        for sentence in sentences:
            if len(sentence) <= 0:
                continue
            tokens = self.tokenizer(sentence)
            if remove_stopwords:
                tokens = [token for token in tokens if token not in self.stopwords]
            tokenized_sentences.append(tokens)
        return tokenized_sentences

    def tag(self, tokenized_sentences):
        """ Tag tokens with the initialized tagger.

        Args:
            tokenized_sentences (List[List[str]]): List of sentences in the form of list of tokens.

        Returns:
            List[List[(str, str)]]: List of sentences in the form of list of (token, tag) pairs.
        """

        # return [self.tagger(tokenized_sentence, tagset='universal') for tokenized_sentence in tokenized_sentences]
        return [self.tagger(tokenized_sentence) for tokenized_sentence in tokenized_sentences]

    def get_np_chunks(self, tokenized_tagged_sentences):
        """ Parse noun-phrase chunks from tokenized-tagged sentences with initialized np-chunk parser.

        Args:
            tokenized_tagged_sentences (List[List[(str, str)]]): List of sentences in the form of list of (token, tag) pairs.

        Returns:
            List[List[(str, str, str)]]: List of noun phrases in the form of list of (token, tag, lemma) pairs.
        """      
  
        np_chunks = [self.np_grammar_parser.parse(sentence) for sentence in tokenized_tagged_sentences]
        sentence_np_chunks = []
        for sentence_chunk in np_chunks:
            sentence_nouns = []
            for token in sentence_chunk:
                if isinstance(token, nltk.tree.Tree) and token._label == "NP":
                    # np_phrase = ' '.join([(token, tag, self.lemmatizer(token)) for token, tag in token.leaves()])
                    # np_phrase = token.leaves()
                    np_phrase = [(token, tag, self.lemmatizer(token).lower()) for token, tag in token.leaves()]
                    sentence_nouns.append(np_phrase)
            sentence_np_chunks.append(sentence_nouns)
        return sentence_np_chunks


    def print_tagged_sentence(self):
        """ Print tagged sentences.
        """

        for sentence in self.sentences:
            print(' '.join(["{}\{}".format(word, tag) for (word, tag, _) in sentence]))


    def print_np_chunks(self):
        """ Print extracted noun-phrase chunks.
        """

        for sentence in self.np_chunks:
            for phrase in sentence:
                print(' '.join([word for (word, _, _) in phrase]))


    def pipeline(self, document):
        """ Pipeline for document preprocessing.

        Args:
            document (str): document to preprocess.
        """

        sentences = self.clean(document)
        tokenized_sentences = self.tokenize(sentences)
        tokenized_tagged_sentences = self.tag(tokenized_sentences)

        self.sentences = [[(word, tag, self.lemmatizer(word).lower()) for (word, tag) in sentence] for sentence in tokenized_tagged_sentences]
        self.np_chunks = self.get_np_chunks(tokenized_tagged_sentences)


if __name__ == '__main__':
    text = "Discrete output feedback sliding mode control of second order systems - a moving switching line approach The sliding mode control systems (SMCS) for which the switching variable is designed independent of the initial conditions are known to be sensitive to parameter variations and extraneous disturbances during the reaching phase. For second order systems this drawback is eliminated by using the moving switching line technique where the switching line is initially designed to pass the initial conditions and is subsequently moved towards a predetermined switching line. In this paper, we make use of the above idea of moving switching line together with the reaching law approach to design a discrete output feedback sliding mode control. The main contributions of this work are such that we do not require to use system states as it makes use of only the output samples for designing the controller. and by using the moving switching line a low sensitivity system is obtained through shortening the reaching phase. Simulation results show that the fast output sampling feedback guarantees sliding motion similar to that obtained using state feedback"
    print(Preprocessing(document=text).sentences)