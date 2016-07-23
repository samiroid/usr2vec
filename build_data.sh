#!/bin/bash -e
clear

 #small|all
 #small: just process a subset of the data
 #all:   use all the data
prepare_mode="all"
#paths to data
user_tweets="DATA/txt/historical_tweets.txt"
clean_user_tweets="DATA/tmp/usrs_history_clean.txt"
#embeddings
embs_txt="DATA/embeddings/sarcasm/sarcasm_embeddings_400.txt"
embs_pkl="DATA/embeddings/sarcasm/sarcasm_embeddings_400.pkl"
#usr2vec_embs="DATA/embeddings/usr2vec_400.pkl"
#usr2vec_embs_txt="DATA/embeddings/usr2vec_400.txt"
#pickles
#aux pickle contains wrd2idx,usr2idx,background_lexical_distribution,E
aux_pickle="DATA/pkl/aux.pkl"
sage_data_path="DATA/pkl/sage_data.pkl"
train_data_path="DATA/pkl/train_data.pkl"
users_per_file=2000
printf "\n#### Preprocess Data ####\n"
python code/prepare_data.py ${user_tweets} ${clean_user_tweets} ${prepare_mode} ${users_per_file}

# printf "\n#### Build Training Data #####\n" 
# python code/build_train.py ${clean_user_tweets} ${embs_txt} ${aux_pickle} ${train_data_path} ${sage_data_path} 

