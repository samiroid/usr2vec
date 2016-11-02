#!/bin/bash -e
clear
train_slice=$1

###########################
# SETUP (edit these paths)
#
# word embeddings
word_embeddings_bin="DATA/embeddings/embeddings_400.pkl"
# user embedding (output)
user_embs_bin="DATA/tmp/usr2vec_400.pkl"
user_embs_txt="DATA/out/usr2vec_400.txt"
if [ ! -z "$1" ]
  then
    user_embs_bin="DATA/tmp/usr2vec_400_"${train_slice}".pkl"
	user_embs_txt="DATA/out/usr2vec_400_"${train_slice}".txt"
fi
#
###########################

###########################
# OPTIONS
#
# number of paralel jobs
n_jobs=5
# number of negative samples
negative_samples=10
#
###########################

### You shouldn't need to change these commands ###
#aux pickle contains wrd2idx,unigram distribution,E
aux_pickle="DATA/tmp/aux.pkl"
usr2idx_path="DATA/tmp/usr2idx"${train_slice}".pkl"
train_data_path="DATA/tmp/train_data"${train_slice}".pkl"
printf "\n##### Estimate Context Conditional Probabilities #####\n"
THEANO_FLAGS="device=cpu" python code/context_probs.py ${train_data_path} ${aux_pickle} ${word_embeddings_bin} ${n_jobs}
printf "\n##### Get Negative Samples #####\n"
THEANO_FLAGS="device=cpu" python code/negative_samples.py ${train_data_path} ${aux_pickle} ${negative_samples} ${n_jobs}
printf "\n##### U2V training #####\n"
python code/train_u2v.py ${train_data_path} ${usr2idx_path} ${aux_pickle} ${user_embs_bin} ${user_embs_txt}
# 