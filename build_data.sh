#!/bin/bash -e
clear

###########################
# SETUP (edit these paths)

#paths to data
user_tweets="DATA/txt/sample.txt"
#embeddings
word_embeddings_txt="DATA/embeddings/embeddings_400.txt"
###########################

###########################
# OPTIONS

#Flag to limit the process to a few examples (e.g., for debug)
#small|all
#small: just process a subset of the data
#all:   use all the data
prepare_mode="small"
#If you are learning vectors from a large number of users, you may want to do the training in parallel by splitting the users into different blocks. If this paramater is > 0, it specifies how many users go into each block (in practice each block is just a file with the tweets from a group of users) 
users_per_file=2000
##########################

# You shouldn't need to chage these settings
clean_user_tweets="DATA/tmp/usrs_history_clean.txt"
#pickles paths
#aux pickle contains wrd2idx,usr2idx,background_lexical_distribution,E
aux_pickle="DATA/tmp/aux.pkl"
train_data_path="DATA/tmp/train_data.pkl"
printf "\n#### Preprocess Data ####\n"
python code/prepare_data.py ${user_tweets} ${clean_user_tweets} ${prepare_mode} ${users_per_file}
printf "\n#### Build Training Data #####\n" 
python code/build_train.py ${clean_user_tweets} ${word_embeddings_txt} ${aux_pickle} ${train_data_path} 

