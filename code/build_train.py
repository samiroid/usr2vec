import argparse
import cPickle
from collections import Counter
from ipdb import set_trace
import numpy as np
import os
import streaming_pickle as stPickle
from contex_probs import ContextProbabilities
np.set_printoptions(threshold=np.nan)
MIN_DOC_LEN=4

def get_parser():
    parser = argparse.ArgumentParser(description="Build Training Data")
    parser.add_argument('-input', type=str, required=True, help='train file(s)')
    parser.add_argument('-emb', type=str, required=True, help='path to word embeddings')
    parser.add_argument('-output', type=str, required=True, help='path of the output')
    parser.add_argument('-vocab_size', type=int, help='path of the output')
    parser.add_argument('-min_docs', type=int, help='reject users with less than min_docs documents',default=10)
    parser.add_argument('-seed', type=int, default=1234, help='random number generator seed')
    parser.add_argument('-db', type=str, required=True, help='word context window scores DB')
    return parser

def window_context_scores(ctx_handler, msg_idx, idx2wrd):
	#convert back to tokens (only the ones that were kept)				
	tokens = [idx2wrd[i] for i in msg_idx]
	#retrieve word-context window scores	
	ctx_score = sum(ctx_handler.score_context_windows(tokens))
	return round(ctx_score,4)			

if __name__ == "__main__" :
	parser = get_parser()
	args = parser.parse_args()	
	rng = np.random.RandomState(args.seed)    

	print "[input: %s | emb: %s | max vocab_size: %d | min_docs: %d | output: %s]" % \
			(os.path.basename(args.input), \
			 os.path.basename(args.emb), \
			 args.vocab_size, \
			 args.min_docs,   \
			 os.path.basename(args.output)) 

	word_counter = Counter()
	n_docs=0	
	with open(args.input,"r") as fid:	
		for line in fid:			
			message = line.decode("utf-8").split()[1:]
			word_counter.update(message)				
			n_docs+=1
	#keep only the args.vocab_size more frequent words
	sw = sorted(word_counter.items(), key=lambda x:x[1],reverse=True)
	top_words = {w[0]:None for w in sw[:args.vocab_size]}
	print "[vocabulary size: %d]" % len(top_words)
	print "loading word embeddings..."
	with open(args.emb) as fid:        
	    # Get emb size
	    vocab_size, emb_size = fid.readline().split()            
	    z=0
	    for line in fid.readlines():      	
	    	z+=1
	    	# if z>=10: break
	        items = line.split()
	        wrd   = items[0]
	        e = np.array(items[1:]).astype(float)            
	        if wrd in top_words:
	        	top_words[wrd] = e        
	if z < int(vocab_size):
		print "out early"
		set_trace()
	
	#generate the embedding matrix
	#keep only words with pre-trained embedding
	words_with_embedding = [i for i,j in top_words.items() if j is not None]
	wrd2idx = {w:i for i,w in enumerate(words_with_embedding)}
	idx2wrd = {i:w for w,i in wrd2idx.items()}
	E = np.zeros((int(emb_size), len(wrd2idx)))   
	for wrd,idx in wrd2idx.items(): E[:, idx] = top_words[wrd]
	print "building training data..."
	if not os.path.exists(os.path.dirname(args.output)):
		os.makedirs(os.path.dirname(args.output))
	#object that retrieves pre-computed word-context window log probability scores
	word_ctx = ContextProbabilities(args.db)
	prev_user, prev_user_data, prev_ctxscores = None, [], []
	wrd_idx_counts = np.zeros(len(wrd2idx))	
	f_train = open(args.output,"wb") 
	rejected_users = []
	with open(args.input,"r") as fid:		
		for j, line in enumerate(fid):					
			message = line.replace("\"", "").replace("'","").split("\t")[1].decode("utf-8").split()	
			#convert to indices
			msg_idx   = [wrd2idx[w] for w in message if w in wrd2idx]			
			if len(msg_idx)<MIN_DOC_LEN: continue				
			u_idx = line.split("\t")[0] 								
			if prev_user is None: 
				#first time 
				prev_user = u_idx
			elif u_idx != prev_user:						
				#after accumulating all documents for current user, shuffle and write them to disk	
				if len(prev_user_data) < args.min_docs:
					#reject users with less than min_docs
					rejected_users.append((u_idx,len(prev_user_data)))
				else:					
					assert len(prev_user_data) == len(prev_ctxscores)
					#shuffle the data			
					shuf_idx = np.arange(len(prev_user_data))
					rng.shuffle(shuf_idx)
					prev_user_data = [prev_user_data[i] for i in shuf_idx]
					prev_ctxscores = [prev_ctxscores[i] for i in shuf_idx]
					# set_trace()					
					split = int(len(prev_user_data)*.9)
					train = prev_user_data[:split]
					test  = prev_user_data[split:]	
					ctx_scores =  prev_ctxscores[:split]
					#each training instance consists of:
					#[user_name, train docs, test documents, word context probabilities, negative samples] 			
					stPickle.s_dump_elt([prev_user, train, test, ctx_scores, [] ], f_train)
					# print "  > user: %s (%d)" % (prev_user, len(train))
				prev_user_data = []
				prev_ctxscores = []							
			elif j == n_docs-1:								
				#reject users with less than min_docs
				if len(prev_user_data) < args.min_docs:
					rejected_users.append((u_idx,len(prev_user_data)))
				else:
					#take into account the very last message	
					prev_user_data.append(msg_idx)				
					#retrieve word-window contex scores
					ctx_score = window_context_scores(word_ctx, msg_idx, idx2wrd)			
					prev_ctxscores.append(ctx_score)
					#shuffle the data			
					shuf_idx = np.arange(len(prev_user_data))
					rng.shuffle(shuf_idx)
					prev_user_data = [prev_user_data[i] for i in shuf_idx]
					prev_ctxscores = [prev_ctxscores[i] for i in shuf_idx]					
					#split
					split = int(len(prev_user_data)*.9)				
					train = prev_user_data[:split]
					test  = prev_user_data[split:]
					ctx_scores =  prev_ctxscores[:split]
					stPickle.s_dump_elt([prev_user, train, test, ctx_scores, [] ], f_train)
					# print "  > user: %s (%d)" % (prev_user, len(train))
			prev_user = u_idx
			prev_user_data.append(msg_idx)
			#retrieve word-window contex scores
			ctx_score = window_context_scores(word_ctx,msg_idx,idx2wrd)			
			prev_ctxscores.append(ctx_score)
			#collect word counts to compute unigram distribution
			for w_idx in msg_idx:								
				wrd_idx_counts[w_idx]+=1	
	f_train.close()				
	print "[rejected users >> %s]" % repr(rejected_users)
	unigram_distribution = wrd_idx_counts / wrd_idx_counts.sum(0)	
	print "[pickling aux data]"
	aux_data = os.path.split(args.output)[0] + "/aux.pkl"
	with open(aux_data,"wb") as fid:
		cPickle.dump([wrd2idx,unigram_distribution, word_counter, E], fid, cPickle.HIGHEST_PROTOCOL)
