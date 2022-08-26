# SIFRank for keyword extraction

SIFRank is an unsupervised keyword extraction algorithm that relies on text embedding using pretrained model from larger corpus. The key idea is to compare the embedding of each candidate key phrase with the embedding of the whole document to find the most similar keywords that best summarize the document. 

## Environment
```
python 3.6
scikit-learn 0.22.2
allennlp 0.8.4
overrides 3.1.0
```


## Usage

```
python SIFRank/sifrank.py
```

ELMO weights: ```elmo_2x4096_512_2048cnn_2xhighway_options.json``` and ```elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5``` can be downloaded [here](https://allenai.org/allennlp/software/elmo) and save to ```models/elmo/``` directory.

## Implementation

### Preprocessing
We perform the standard preprocessing procedure on the document with nltk, including character cleaning, tokenizing, pos-tagging, lemmatizing and noun-phrase chunk extraction using fixed grammar. Note that sentence splitting is not necessary for keyword extraction as the embedding of the whole document is computed for comparison.

### Embedding computation
Word weights are initialized with vocabulary extracted from English wikipedia: ```models/enwiki_vocab_min200.txt```.
The embeddor we select to use for now is ELMO. 

<!-- ### Graph construction
We used the networkx package for graph operations. Each token that pass the POS-tag filter is added to the graph as a node. For each token within a window size range of a given token in the same sentence, an edge is added with initial weight 1, or incremented by 1 for existing edge between the two corresponding nodes. 

### PageRank
Given a weighted undirected graph, we perform the PageRank algorithm by considering two directed edges with same weight for each undirected edge. The modified PageRank algorithm, taking weights into account, computes a score for each keyword node with the following formula. 

$WS(V_i) = (1 - d) + d \times \sum_{V_j \in In(V_i)} \frac{w_{ji}}{\sum_{V_k \in Out(V_j)} w_{jk}} WS(V_j)$

where $In(V)$ is the set of vertices that points to vertex $V$ and $Out(V)$ the vertices that $V$ points to. The damping factor $d$, between $0$ and $1$, denotes the probability of jumping to another vertex in the original PageRank algorithm of the web-surfing context. Convergence is achieved when the score update for any node from the previous iteration is smaller than a threshold. The top-T (usually one-third of all nodes) keywords are considered candiates for the post-processing phase.


### Post-processing
From single keywords to meaningful key phrases, TextRank proposed to aggregrate adjacent keywords into phrases. In the implementation we greedily select adjacent candiate keywords and sum their scores. 


One limitation we noticed is that the default algorithm could not extract meaningful entity names (e.g. lyrics of *shape of you* from the dataset) by greedily combining adjacent keywords. We referred to the implementation of pytextrank and used its modified post-processing technique by evaluating ranks for meaningful entities extracted in the early preprocessing step. -->

## Example

> Compatibility of systems of linear constraints over the set of natural numbers. Criteria of compatibility of a system of linear Diophantine equations, strict inequations, and nonstrict inequations are considered. Upper bounds for components of a minimal set of solutions and algorithms of construction of minimal generating sets of solutions for all types of systems are given. These criteria and the corresponding algorithms for constructing a minimal supporting set of solutions can be used in solving all the considered types systems and systems of mixed types.

We perform SIFRank on the above text from the TextRank paper. 

<!-- Preprocessing gives the following tagged tokens and noun-phrases:
```
Compatibility\NN of\IN systems\NNS of\IN linear\JJ constraints\NNS over\IN the\DT set\NN of\IN natural\JJ numbers\NNS
Criteria\NNS of\IN compatibility\NN of\IN a\DT system\NN of\IN linear\JJ Diophantine\NNP equations\NNS ,\, strict\JJ inequations\NNS ,\, and\CC nonstrict\JJ inequations\NNS are\VBP considered\VBN
Upper\NNP bounds\VBZ for\IN components\NNS of\IN a\DT minimal\JJ set\NN of\IN solutions\NNS and\CC algorithms\NN of\IN construction\NN of\IN minimal\JJ generating\VBG sets\NNS of\IN solutions\NNS for\IN all\DT types\NNS of\IN systems\NNS are\VBP given\VBN
These\DT criteria\NNS and\CC the\DT corresponding\JJ algorithms\NN for\IN constructing\VBG a\DT minimal\JJ supporting\NN set\NN of\IN solutions\NNS can\MD be\VB used\VBN in\IN solving\VBG all\PDT the\DT considered\VBN types\NNS systems\NNS and\CC systems\NNS of\IN mixed\JJ types\NNS .\.
``` -->

Computing similarity of phrase embeddings and document embedding gives the following keywords that best summarize the document.

```
0.90793: switching line technique
0.90002: sliding mode control system SMCS
0.89852: predetermined switching line
0.89791: discrete output feedback
0.87867: switching line
0.86175: mode control
0.85978: low sensitivity system
0.85822: switching variable
0.83122: phase Simulation result
0.82286: extraneous disturbance
0.82159: Discrete output
0.82133: second order system
0.82039: fast output
0.81772: output sample
0.81478: feedback guarantee
0.81011: law approach
0.80955: mode
0.80401: main contribution
0.79626: initial condition
0.78824: line
```