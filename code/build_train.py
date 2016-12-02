import argparse
import cPickle
from collections import Counter
from ipdb import set_trace
import numpy as np
import sys
import streaming_pickle as stPickle
import os

rng = np.random.RandomState(1234)     
np.set_printoptions(threshold=np.nan)
MIN_MSG_SIZE=4

def get_parser():
    parser = argparse.ArgumentParser(description="Build Training Data")
    parser.add_argument('-input', type=str, required=True, help='train file(s)')
    parser.add_argument('-emb', type=str, required=True, help='path to word embeddings')
    parser.add_argument('-output', type=str, required=True, help='path of the output')
    parser.add_argument('-vocab_size', type=int, help='path of the output')
    parser.add_argument('-min_docs', type=int, help='reject users with less than min_docs documents',default=10)
    return parser

if __name__ == "__main__" :
	parser = get_parser()
	args = parser.parse_args()	
	# docs, emb_path, training_data_path, aux_data = sys.argv[1:]
	print "[input: %s | emb: %s | max vocab_size: %d | min_docs: %d |output: %s]" % \
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
	E = np.zeros((int(emb_size), len(wrd2idx)))   
	for wrd,idx in wrd2idx.items(): E[:, idx] = top_words[wrd]
	print "building training data..."
	prev_user, prev_user_data = None, []
	word_counts = np.zeros(len(wrd2idx))
	f_train = open(args.output,"wb") 
	rejected_users = []
	with open(args.input,"r") as fid:		
		for j, line in enumerate(fid):					
			message = line.split("\t")[1].decode("utf-8").split()	
			#convert to indices
			msg_idx = [wrd2idx[w] for w in message if w in wrd2idx]
			if len(msg_idx)<MIN_MSG_SIZE: continue					
			u_idx = line.split("\t")[0] 					
			#after accumulating all documents for current user, shuffle and write them to disk		
			if prev_user is None:
				#first time 
				prev_user = u_idx
			elif u_idx != prev_user:						
				#reject users with less than min_docs
				if len(prev_user_data) < args.min_docs:
					rejected_users.append((u_idx,len(prev_user_data)))
				else:
					#shuffle the data					
					rng.shuffle(prev_user_data)
					split = int(len(prev_user_data)*.9)
					train = prev_user_data[:split]
					test  = prev_user_data[split:]	
					#each training instance consists of:
					#[user_name, train docs, test documents, word context probabilities, negative samples] 			
					stPickle.s_dump_elt([prev_user, train, test, [],[]], f_train)
					print "  > user: %s (%d)" % (prev_user, len(train))
				prev_user_data = []							
			elif j == n_docs-1:				
				#take into account the very last message	
				prev_user_data.append(msg_idx)				
				#reject users with less than min_docs
				if len(prev_user_data) < args.min_docs:
					rejected_users.append((u_idx,len(prev_user_data)))
				else:
					#shuffle the data		
					rng.shuffle(prev_user_data)
					split = int(len(prev_user_data)*.9)				
					train = prev_user_data[:split]
					test  = prev_user_data[split:]
					stPickle.s_dump_elt([prev_user, train, test, [],[]], f_train)
					print "  > user: %s (%d)" % (prev_user, len(train))
			prev_user = u_idx
			prev_user_data.append(msg_idx)
			#collect word counts to compute unigram distribution
			for w_idx in msg_idx:								
				word_counts[w_idx]+=1	
	f_train.close()				
	print "[rejected users >> %s]" % repr(rejected_users)
	unigram_distribution = word_counts / word_counts.sum(0)	
	print "[pickling aux data]"
	aux_data = os.path.split(args.output)[0] + "/aux.pkl"
	with open(aux_data,"wb") as fid:
		cPickle.dump([wrd2idx,unigram_distribution,E], fid, cPickle.HIGHEST_PROTOCOL)
