#!/bin/bash -e
clear
train_slice=$1

###########################
# SETUP (edit these paths)
#
# word embeddings
WORD_EMBEDDINGS_BIN="DATA/embeddings/embs_emoji_2_400"
# user embedding (output)
#user_embs_bin="DATA/tmp/usr2vec_400.pkl"
#user_embs_txt="DATA/out/usr2vec_400.txt"
#if [ ! -z "$1" ]
#  then
#    user_embs_bin="DATA/tmp/usr2vec_400_"${train_slice}".pkl"
#	user_embs_txt="DATA/out/usr2vec_400_"${train_slice}".txt"
#fi
#
###########################

###########################
# OPTIONS
#
# number of paralel jobs
N_JOBS=17
# number of negative samples
negative_samples=10
#
###########################

### You shouldn't need to change these commands ###
#aux pickle was generated automatically when building the training data
# contains wrd2idx,unigram distribution,E
aux_pickle="DATA/tmp/aux.pkl"
train_data_path="DATA/tmp/train_data"${train_slice}".pkl"
printf "\n##### Estimate Context Conditional Probabilities #####\n"
THEANO_FLAGS="device=cpu" python code/context_probs.py ${train_data_path} ${aux_pickle} ${WORD_EMBEDDINGS_BIN} ${N_JOBS}
printf "\n##### Get Negative Samples #####\n"
THEANO_FLAGS="device=cpu" python code/negative_samples.py ${train_data_path} ${aux_pickle} ${negative_samples} ${N_JOBS}
# 
