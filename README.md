Usr2Vec
=======

Implementation of the the *usr2vec* model to induce neural user embeddings, as described in the paper in the [paper](https://arxiv.org/abs/1705.00335) *Quantifying Mental Health from Social Media with Neural User Embeddings*. The resulting embeddings capture latent user aspects, e.g. political leanings and mental-health status (Figure 1)

A previous version of this model, described in the paper *Modelling Context with User Embeddings for Sarcasm Detection in Social Media* can be found [here](https://github.com/samiroid/usr2vec/tree/v1).

![alt text](https://i.imgur.com/hbrY4bU.jpg "User Embeddings") Figure 1 - User embeddings projected into 2-Dimensions and colored according to mental health status. 

If you use this code please cite our paper as:
> Amir, S., Coppersmith, G., Carvalho, P., Silva, M.J. and Wallace, B.C., 2017. *Quantifying Mental Health from Social Media with Neural User Embeddings*. In Journal of Machine Learning Research, W&C Track, Volume 68. 

## Requirements:
The software is implemented in python 2.7 and requires the following packages:
* [sma_toolkit](https://github.com/samiroid/sma_toolkit)
* numpy
* gensim
* joblib
* theano

### Inputs/Outputs:

There are two inputs to this model: 
1. a text file with the training data --- the system assumes that the documents can be tokenized using whitespace (we recommend pre-tokenzing with the appropriate tokenizer) and that all messages from a user appear sequentially (see `raw_data/sample.txt` for an example)
2. a text file with pre-trained word embeddings (e.g. word2vec, glove)

The output is a text file with a format similar to word2vec's, i.e. each line consists of `user_id \t embedding`.

## Instructions
The software works in two main steps: (1) building the training the data; and (2) learning the user embeddings. The code can be executed as follow:

1. Setup 
    1. get the *sma_toolkit*
    2. edit `scripts/setup.sh` and set the path to the *sma_toolkit*
    3. run `./scripts/setup.sh` 
2. Build training data
    1. get some pretrained word embeddings
    2. edit the paths on the file `scripts/build_data.sh` (i.e. the variables *DATA_PATH*, *WORD_EMBEDDINGS*)
    3. run `./scripts/build_data.sh` 
3. Train model: run `./scripts/build_data.sh [DATA_PATH] [OUTPUT_PATH]`

