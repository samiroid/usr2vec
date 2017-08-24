#!/bin/bash -e
# clear

train_data_path=$1".pkl"
aux_pickle=$1"_aux.pkl"
user_embs_txt=$2

lrate=0.00010
printf "\n##### U2V training #####\n"
python code/train_u2v.py -input ${train_data_path} \
								-aux ${aux_pickle} \
								-output ${user_embs_txt} \
						 		-patience 5 \
						 		-margin 1 \
						 		-epochs 12 \
						 		-lrate $lrate
						 


