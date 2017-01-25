#!/bin/bash -e
# clear
train_slice=$1

###########################
# SETUP (edit these paths)
#
# user embedding (output)
user_embs_bin="DATA/pkl/usr2vec_twmh.pkl"
user_embs_txt="DATA/out/usr2vec_twmh.txt"
if [ ! -z "$1" ]
  then
    user_embs_bin="DATA/pkl/usr2vec_twmh_"${train_slice}".pkl"
	user_embs_txt="DATA/out/usr2vec_twmh_"${train_slice}".txt"
fi
#
###########################

### You shouldn't need to change these commands ###
#aux pickle contains wrd2idx,unigram distribution and word embedding matrix
aux_pickle="DATA/pkl/aux.pkl"
train_data_path="DATA/pkl/train_data"${train_slice}".pkl"

printf "\n##### U2V training #####\n"
python code/train_u2v.py -input ${train_data_path} -aux ${aux_pickle} -output ${user_embs_txt} \
						 -patience 5 \
						 -margin 1 \
						 -epochs 25 \
						  -reshuff
						 #-lrate 0.0002 \
						 # -init_w2v mean
