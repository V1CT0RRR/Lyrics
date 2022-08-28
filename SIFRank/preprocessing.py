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
                 sent_tokenizer=nltk.tokenize.sent_tokenize,
                 tokenizer=nltk.tokenize.word_tokenize,
                 tagger=nltk.tag.pos_tag,
                 np_grammar=GRAMMAR_3_NLTK,
                 nlp=None):

        self.stopwords = set(nltk.corpus.stopwords.words("english"))

        self.tokenizer = tokenizer
        self.sent_tokenizer = sent_tokenizer
        self.tagger = tagger

        # https://www.nltk.org/book_1ed/ch07.html
        # https://github.com/sunyilgdx/SIFRank/blob/274d84b797c449e66c414d887f15d9b40114c746/model/extractor.py#L10
        self.np_grammar = np_grammar 
        self.np_grammar_parser = nltk.RegexpParser(self.np_grammar)

        self.nlp = nlp

        self.pipeline(document)


    def clean(self, document):
        sentences = nltk.tokenize.sent_tokenize(document.replace('\n', '. '))
        for idx, sentence in enumerate(sentences):
            sentences[idx] = re.sub("[^a-zA-Z0-9'.,\- ]", '', sentence)#.lower()
        return sentences


    def tokenize(self, sentences, remove_stopwords=False):
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
        return [self.tagger(tokenized_sentence) for tokenized_sentence in tokenized_sentences]


    def get_np_chunks(self, tagged_tokens):
        np_chunks = self.np_grammar_parser.parse(tagged_tokens)
        candidates = []
        for token in np_chunks:
            if isinstance(token, nltk.tree.Tree) and token._label == "NP":
                np_phrase = ' '.join([token for token, _, _ in token.leaves()])
                _, _, start = token.leaves()[0]
                end = start + len(token.leaves())
                candidates.append((np_phrase, (start, end)))
        return candidates


    def pipeline(self, document):

        if self.nlp is None:
            sentences = self.clean(document)
            self.tokenized_sentences = self.tokenize(sentences)
            self.tagged_tokenized_sentences = self.tag(self.tokenized_sentences)

            self.tokens = [token for sentence in self.tokenized_sentences for token in sentence]
            self.tagged_tokens = [tagged_token for sentence in self.tagged_tokenized_sentences for tagged_token in sentence]
            self.tagged_tokens = [(token, tag, idx) for idx, (token, tag) in enumerate(self.tagged_tokens)]

            self.np_candidates = self.get_np_chunks(self.tagged_tokens)

        else:
            spacy_doc = self.nlp(document.lower())

            self.tokenized_sentences = [[spacy_doc[token_idx].text for token_idx in range(sent.start, sent.end)] for sent in spacy_doc.sents]
            self.tagged_tokenized_sentences = [[(spacy_doc[token_idx].text, spacy_doc[token_idx].pos_) for token_idx in range(sent.start, sent.end)] for sent in spacy_doc.sents]

            self.tokens = [token for sentence in self.tokenized_sentences for token in sentence]
            self.tagged_tokens = [tagged_token for sentence in self.tagged_tokenized_sentences for tagged_token in sentence]
            self.tagged_tokens = [(token, tag, idx) for idx, (token, tag) in enumerate(self.tagged_tokens)]

            self.np_candidates = [(noun_chunk.text, (noun_chunk.start, noun_chunk.end)) for noun_chunk in spacy_doc.noun_chunks]


    def show_candidates(self):
        for (np_phrase, (start, end)) in self.np_candidates:
            print("{:<4} - {:<4}: {}".format(start, end, np_phrase))

if __name__ == '__main__':
    text = "Discrete output feedback sliding mode control of second order systems - a moving switching line approach The sliding mode control systems (SMCS) for which the switching variable is designed independent of the initial conditions are known to be sensitive to parameter variations and extraneous disturbances during the reaching phase. For second order systems this drawback is eliminated by using the moving switching line technique where the switching line is initially designed to pass the initial conditions and is subsequently moved towards a predetermined switching line. In this paper, we make use of the above idea of moving switching line together with the reaching law approach to design a discrete output feedback sliding mode control. The main contributions of this work are such that we do not require to use system states as it makes use of only the output samples for designing the controller. and by using the moving switching line a low sensitivity system is obtained through shortening the reaching phase. Simulation results show that the fast output sampling feedback guarantees sliding motion similar to that obtained using state feedback"
    Preprocessing(document=text).show_candidates()