#!/bin/bash -e
clear

 #small|all
 #small: just process a subset of the data
 #all:   use all the data
prepare_mode="small"
#paths to data
user_tweets="DATA/txt/historical_tweets.txt"
clean_user_tweets="DATA/tmp/usrs_history_clean.txt"
#embeddings
word_embeddings_txt="DATA/embeddings/sarcasm/sarcasm_embeddings_400.txt"
#pickles
#aux pickle contains wrd2idx,usr2idx,background_lexical_distribution,E
aux_pickle="DATA/pkl/aux.pkl"
train_data_path="DATA/pkl/train_data.pkl"
#this parameter can be used set to process users in parallel
users_per_file=2000
printf "\n#### Preprocess Data ####\n"
python code/prepare_data.py ${user_tweets} ${clean_user_tweets} ${prepare_mode} ${users_per_file}
printf "\n#### Build Training Data #####\n" 
python code/build_train.py ${clean_user_tweets} ${word_embeddings_txt} ${aux_pickle} ${train_data_path} 

