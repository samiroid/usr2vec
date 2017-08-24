#!/bin/bash -e
# clear
train_slice=$1
lrate=0.00010

###########################
# SETUP (edit these paths)
#
# user embedding (output)
user_embs_txt="DATA/out/tmh_u2v.txt"

if [ ! -z "$1" ]
  then   
	user_embs_txt="DATA/out/usr2vec_twmh_"${train_slice}".txt"
fi
# if [ -z "$2" ]
#   then   
# 	echo "missing lrate"
# 	exit
# fi
#
###########################

### You shouldn't need to change these commands ###
#aux pickle contains wrd2idx,unigram distribution and word embedding matrix
aux_pickle="DATA/pkl/aux.pkl"
train_data_path="DATA/pkl/train_data"${train_slice}".pkl"

printf "\n##### U2V training #####\n"
python code/train_u2v_byuser.py -input ${train_data_path} -aux ${aux_pickle} -output ${user_embs_txt} \
						 -patience 5 \
						 -margin 1 \
						 -epochs 12 \
						 -lrate $lrate \
						 -nocp \
						 -quiet
						 # -init_w2v mean
