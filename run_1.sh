#!/bin/bash -e
clear

n_jobs=4
#embeddings
embs_txt="DATA/embeddings/sarcasm/sarcasm_embeddings_400.txt"
embs_pkl="DATA/embeddings/sarcasm/sarcasm_embeddings_400.pkl"
#paths to data
clean_user_tweets="DATA/tmp/usrs_history_clean_1.txt"
usr2vec_embs="DATA/embeddings/usr2vec_400_1.pkl"
usr2vec_embs_txt="DATA/embeddings/usr2vec_400_1.txt"
aux_pickle="DATA/pkl/aux_1.pkl"
sage_data_path="DATA/pkl/sage_data_1.pkl"
sage_params="DATA/pkl/sage_etas_1.pkl"
train_data_path="DATA/pkl/train_data_1.pkl"
# u2v_train_data_path="DATA/pkl/u2v_train_data.pkl"


 printf "\n#### Build Training Data #####\n" 
 python code/build_train.py ${clean_user_tweets} ${embs_txt} ${aux_pickle} ${train_data_path} ${sage_data_path} 

printf "\n##### Estimate Context Conditional Probabilities #####\n"
 THEANO_FLAGS="device=cpu" python code/extract_logprobs.py ${aux_pickle} ${embs_pkl} ${train_data_path} ${n_jobs}

 printf "\n##### SAGE fitting #####\n"
 THEANO_FLAGS="device=cpu" python code/train_sage.py ${aux_pickle} ${sage_data_path} ${sage_params} ${n_jobs}

 printf "\n##### Get Negative Samples #####\n"
 THEANO_FLAGS="device=cpu" python code/negative_samples.py ${aux_pickle}  ${sage_params} ${train_data_path} ${n_jobs}

echo "\n##### U2V training #####\n"
python code/gpu_train_u2v.py ${train_data_path} ${aux_pickle} ${usr2vec_embs} #${n_jobs}

echo "\n##### Exporting #####\n"
python code/export_embeddings.py ${aux_pickle} ${usr2vec_embs} ${usr2vec_embs_txt} 
