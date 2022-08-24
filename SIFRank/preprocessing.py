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
    def __init__(self, document, tokenizer=None, tagger=None, np_grammar=None):

        self.stopwords = set(nltk.corpus.stopwords.words("english"))

        self.tokenizer = nltk.tokenize.word_tokenize if not tokenizer else tokenizer
        self.tagger = nltk.tag.pos_tag if not tagger else tagger

        # https://www.nltk.org/book_1ed/ch07.html
        # https://github.com/sunyilgdx/SIFRank/blob/274d84b797c449e66c414d887f15d9b40114c746/model/extractor.py#L10
        self.np_grammar = GRAMMAR_3_NLTK if not np_grammar else np_grammar 
        self.np_grammar_parser = nltk.RegexpParser(self.np_grammar)

        self.pipeline(document)

    def clean(self, document):
        document = re.sub("[^a-zA-Z0-9' ]", '', document)#.lower()
        return document

    def tokenize(self, text, remove_stopwords=True):
        tokens = self.tokenizer(text)
        if remove_stopwords:
            return [token for token in tokens if token not in self.stopwords]
        return tokens

    def tag(self, tokens):
        return self.tagger(tokens)

    def get_np_chunks(self, tokenized_tagged_sentences):
        np_chunks = self.np_grammar_parser.parse(tokenized_tagged_sentences)
        candidates = []
        for token in np_chunks:
            if isinstance(token, nltk.tree.Tree) and token._label == "NP":
                np_phrase = ' '.join([token for token, _, _ in token.leaves()])
                _, _, start = token.leaves()[0]
                end = start + len(token.leaves())
                candidates.append((np_phrase, (start, end)))
        return candidates

    def pipeline(self, document):

        '''
        Step 1: The document is tokenized and part-of-speech
        tagged to sequence of tokens with part-of-speech tags.
        '''
        document = self.clean(document)
        self.tokenized_sentences = self.tokenize(document, remove_stopwords=False)
        self.tokenized_tagged_sentences = self.tag(self.tokenized_sentences)

        self.tokenized_tagged_sentences = [(token, tag, idx) for idx, (token, tag) in enumerate(self.tokenized_tagged_sentences)]

        '''
        Step 2: Extract the noun phrases (NPs) from the sequence 
        according to the part-of-speech tags using NP-chunker 
        (pattern wrote by regular expression). The NPs extracted 
        from the document are the candidate keyphrases.
        '''
        self.np_candidates = self.get_np_chunks(self.tokenized_tagged_sentences)

        return self.np_candidates

    def show_candidates(self):
        for (np_phrase, (start, end)) in self.np_candidates:
            print("{:<4} - {:<4}: {}".format(start, end, np_phrase))

if __name__ == '__main__':
    text = "Discrete output feedback sliding mode control of second order systems - a moving switching line approach The sliding mode control systems (SMCS) for which the switching variable is designed independent of the initial conditions are known to be sensitive to parameter variations and extraneous disturbances during the reaching phase. For second order systems this drawback is eliminated by using the moving switching line technique where the switching line is initially designed to pass the initial conditions and is subsequently moved towards a predetermined switching line. In this paper, we make use of the above idea of moving switching line together with the reaching law approach to design a discrete output feedback sliding mode control. The main contributions of this work are such that we do not require to use system states as it makes use of only the output samples for designing the controller. and by using the moving switching line a low sensitivity system is obtained through shortening the reaching phase. Simulation results show that the fast output sampling feedback guarantees sliding motion similar to that obtained using state feedback"
    Preprocessing(document=text).show_candidates()