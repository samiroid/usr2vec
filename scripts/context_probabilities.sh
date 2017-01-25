#!/bin/bash -e
# clear

WORD_EMBEDDINGS_BIN="DATA/embeddings/bin/word2vec.pkl"
DATA="raw_data/sample.txt"
CTX_PROBS_DB="raw_data/twmh_ctxprobs.db"

python code/ctx_window_probs.py -input $DATA \
										-emb ${WORD_EMBEDDINGS_BIN} \
										-db $CTX_PROBS_DB \
										-mode "extract" "score" \
										-overwrite
													   											   

