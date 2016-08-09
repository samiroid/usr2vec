#!/bin/bash -e
clear

n_jobs=20
#embeddings
word_embeddings_bin="DATA/embeddings/sarcasm/sarcasm_embeddings_400.pkl"
#paths to data
usr2vec_embs="DATA/embeddings/usr2vec_400.pkl"
usr2vec_embs_txt="DATA/embeddings/usr2vec_400.txt"
aux_pickle="DATA/pkl/aux.pkl"
train_data_path="DATA/pkl/train_data.pkl"
printf "\n##### Estimate Context Conditional Probabilities #####\n"
THEANO_FLAGS="device=cpu" python code/extract_logprobs.py ${aux_pickle} ${word_embeddings_bin} ${train_data_path} ${n_jobs}
printf "\n##### Get Negative Samples #####\n"
THEANO_FLAGS="device=cpu" python code/negative_samples.py ${aux_pickle}   ${train_data_path} ${n_jobs} 
printf "\n##### U2V training #####\n"
python code/gpu_train_u2v.py ${train_data_path} ${aux_pickle} ${usr2vec_embs} #${n_jobs}
printf "\n##### Exporting #####\n"
python code/export_embeddings.py ${aux_pickle} ${usr2vec_embs} ${usr2vec_embs_txt} 
