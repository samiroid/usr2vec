#!/bin/bash -e
###########################
# SETUP (edit these paths)
###########################
#
# *** USER DOCUMENTS ***
# IMPORTANT the system assumes the documents:
# 1. can be white-space tokenized 
# 2. are sorted by user (i.e., all the documents of a given user appear sequentially in the file)
# 3. have at least MIN_MSG_SIZE=4 words (see build_data.py)
DATA="raw_data/sample.txt"
#
# *** WORD EMBEDDINGS ***
WORD_EMBEDDINGS_TXT="/Users/samir/Dev/resources/embeddings/word_embedddings.txt"
#
# *** OUTPUT ***
OUTPUT_PATH="DATA/pkl/sample.pkl"
#
# *** OPTIONS ***
MAX_VOCAB_SIZE=50000
MIN_DOCS=100 #reject users with less than this number of documents
NEGATIVE_SAMPLES=20
#
##########################

#When trying to learn embeddings for a large number of users, one may want to parallelize training by splitting the users into different blocks. 
#`n_splits' specifies the number of partitions of the training data (1 is the default)
n_splits=1
if [ ! -z "$1" ]
  then
    n_splits=$1      
    echo "### n_splits:"$n_splits
fi
clear
printf "cleaning up...\n"


### ACTION!
printf "\n#### Build Training Data #####\n"
THEANO_FLAGS="device=cpu" python code/build_train.py -input ${DATA} -emb ${WORD_EMBEDDINGS_TXT} \
													-output ${OUTPUT_PATH} \
													-vocab_size ${MAX_VOCAB_SIZE} \
													-neg_samples $NEGATIVE_SAMPLES

# Split training data by number of user documents
if (($n_splits > 1 )); 
	then
		printf "\n#### Sort and Split Training Data #####\n"
		python code/sort_split.py -input ${OUTPUT_PATH} -n_splits ${n_splits} 
fi
