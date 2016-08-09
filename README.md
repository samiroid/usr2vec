Usr2Vec
=======

Code for learning user representations as described in the paper *Modelling Context with User Embeddings for Sarcasm Detection in Social Media* [[paper] (https://arxiv.org/abs/1607.00976)] [[bib] ()]

pre-requisites:

* my_utils -> https://github.com/samiroid/utils

requirements:
* python >= 2.7
* numpy
* gensim
* joblib
* theano


## Running the code

0. pretrain word embeddings using [gensim] (https://radimrehurek.com/gensim/models/word2vec.html) with hierarchical softmax (see the documention on how to do this). Save the embeddings in binary format.  
1. clone or download the [my_utils] (https://github.com/samiroid/utils) module and place it under the folder `code`
2. 
