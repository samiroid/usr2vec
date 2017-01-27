#!/bin/bash -e
# clear

WORD_EMBEDDINGS_BIN="DATA/embeddings/bin/embs_emoji_2_400"
#DATA="raw_data/sample.txt"
DATA="raw_data/user_corpus.txt"
# CTX_PROBS_DB="raw_data/twmh_ctxprobs.db"
CTX_PROBS_DB="/home/ubuntu/twmh_ctxprobs.db"

python code/ctx_window_probs.py -input $DATA \
										-emb ${WORD_EMBEDDINGS_BIN} \
										-db $CTX_PROBS_DB \
										-mode "extract" "score" \
										-overwrite
													   											   

