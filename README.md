Usr2Vec
=======

Code for learning user representations as described in the paper *Modelling Context with User Embeddings for Sarcasm Detection in Social Media* [[paper] (https://arxiv.org/abs/1607.00976)] [[bib] ()]

requirements:
* python >= 2.7
* my_utils (grab it [here] (https://github.com/samiroid/utils))
* numpy
* gensim
* joblib
* theano


## Running the code

0. pretrain word embeddings using [gensim] (https://radimrehurek.com/gensim/models/word2vec.html) with the **hierarchical softmax** option (see the [documention] (https://radimrehurek.com/gensim/models/word2vec.html) on how to do this--tl;dr set the flag *hs=1*). Save the embeddings in binary format.  
1. clone or download the [my_utils] (https://github.com/samiroid/utils) module
2. edit file `setup.sh` to change the paths to `my_utils` and the word embeddings; run `setup.sh`
3. edit file `build_data.sh` to change the paths to the word embeddings and the file containing the user's tweets; run `build_data.sh`
4. edit file `run.sh` to change the paths to the word embeddings (binary format) and the ouput user embeddings; run `run.sh`
5. kick-back and relax :)
