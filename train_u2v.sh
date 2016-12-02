#!/bin/bash -e
# clear
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

### You shouldn't need to change these commands ###
#aux pickle contains wrd2idx,unigram distribution and word embedding matrix
aux_pickle="DATA/tmp/aux.pkl"
train_data_path="DATA/tmp/train_data"${train_slice}".pkl"

printf "\n##### U2V training #####\n"
# python code/train_u2v.py ${train_data_path} ${aux_pickle} ${user_embs_bin} ${user_embs_txt}


# -input INPUT -aux AUX [-output OUTPUT] [-lrate LRATE]
#                     [-margin MARGIN] [-epochs EPOCHS] [-patience PATIENCE]

python code/train_u2v.py -input ${train_data_path} -aux ${aux_pickle} -output ${user_embs_txt} \
						 -patience 5 \
						 -margin 1 \
						 -epochs 25 \
						 -lrate 0.0002 \
						 # -init_w2v mean
						 # -reshuff

