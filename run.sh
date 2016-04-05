#!/bin/bash -e
clear

 #small|all
 #small: just process a subset of the data
 #all:   use all the data
prepare_mode="small"
#paths to data
user_tweets="DATA/txt/usrs_history_trial.txt"
clean_user_tweets="DATA/txt/usrs_history_clean.txt"
#embeddings
embs_txt="DATA/embeddings/sarcasm/sarcasm_embeddings_400.txt"
embs_pkl="DATA/embeddings/sarcasm/sarcasm_embeddings_400.pkl"
usr2vec_embs="DATA/embeddings/usr2vec_400.pkl"
#pickles
#aux pickle contains wrd2idx,usr2idx,background_lexical_distribution,E
aux_pickle="DATA/pkl/aux.pkl"
sage_data_path="DATA/pkl/sage_data.pkl"
sage_params="DATA/pkl/sage_etas.pkl"
train_data_path="DATA/pkl/train_data.pkl"
# u2v_train_data_path="DATA/pkl/u2v_train_data.pkl"
n_jobs=8

# printf "\n#### Preprocess Data ####\n"
# python code/prepare_data.py ${user_tweets} ${clean_user_tweets} ${prepare_mode}

# printf "\n#### Build Training Data #####\n" 
# python code/build_train.py ${clean_user_tweets} ${embs_txt} ${aux_pickle} ${train_data_path} ${sage_data_path} 

printf "\n##### SAGE fitting #####\n"
python code/train_sage.py ${aux_pickle} ${sage_data_path} ${sage_params} ${n_jobs}

# echo "\n##### Extract #####\n"
# python code/extract.py ${aux_pickle} ${embs_pkl} ${sage_params} ${train_data_path} ${n_jobs}

# echo "\n##### U2V training #####\n"
# python code/train_u2v.py ${train_data_path} ${aux_pickle} ${usr2vec_embs} ${n_jobs}