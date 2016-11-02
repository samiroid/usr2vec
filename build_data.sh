#!/bin/bash -e
clear
printf "cleaning up...\n"
rm DATA/tmp/*.* || true
rm DATA/out/*.* || true


###########################
# SETUP (edit these paths)
#
#paths to data
user_tweets="DATA/txt/historical_tweets.txt"
#embeddings
word_embeddings_txt="DATA/embeddings/embeddings_400.txt"
#
###########################

###########################
# OPTIONS
#
#Flag to limit the process to a few examples (e.g., for debug)
#small|all
#small: just process a subset of the data
#all:   use all the data
prepare_mode="small"
#When trying to learning embeddings for a large number of users, one may want to parallelize training by splitting the users into different blocks. `n_slices' specifies the number of partitions of the training data (1 is the default)
n_slices=1
if [ ! -z "$1" ]
  then
    n_slices=$1
fi
##########################
# You shouldn't need to chage these settings
clean_tweets="DATA/tmp/clean_tweets.txt"
#pickles paths
#aux pickle contains wrd2idx,unigram_distribution,E
aux_pickle="DATA/tmp/aux.pkl"
usr2idx="DATA/tmp/usr2idx.pkl"
train_data_path="DATA/tmp/train_data.pkl"
printf "\n#### Preprocess Data ####\n"
python code/prepare_data.py ${user_tweets} ${clean_tweets} ${prepare_mode}
printf "\n#### Build Training Data #####\n"
python code/build_train.py ${clean_tweets} ${word_embeddings_txt} ${train_data_path} ${aux_pickle} ${usr2idx} ${n_slices}