#!/bin/bash -e
clear
train_slice=$1

###########################
# SETUP (edit these paths)
#
# word embeddings
word_embeddings_bin="DATA/embeddings/embeddings_400.pkl"
# user embedding (output)
usr2vec_embs="DATA/tmp/usr2vec_400.pkl"
usr2vec_embs_txt="DATA/out/usr2vec_400_"${train_slice}".txt"
#
###########################

###########################
# OPTIONS
#
# number of paralel jobs
n_jobs=2
#
###########################

### You shouldn't need to change these commands ###
#aux pickle contains wrd2idx,usr2idx,background_lexical_distribution,E
aux_pickle="DATA/tmp/aux"${train_slice}".pkl"
train_data_path="DATA/tmp/train_data"${train_slice}".pkl"
printf "\n##### Estimate Context Conditional Probabilities #####\n"
THEANO_FLAGS="device=cpu" python code/context_logprobs.py ${train_data_path} ${aux_pickle} ${word_embeddings_bin} ${n_jobs}
printf "\n##### Get Negative Samples #####\n"
THEANO_FLAGS="device=cpu" python code/negative_samples.py ${aux_pickle}   ${train_data_path} ${n_jobs} 
printf "\n##### U2V training #####\n"
python code/gpu_train_u2v.py ${train_data_path} ${aux_pickle} ${usr2vec_embs} #${n_jobs}
printf "\n##### Exporting #####\n"
python code/export_embeddings.py ${aux_pickle} ${usr2vec_embs} ${usr2vec_embs_txt} 
