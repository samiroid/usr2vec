import argparse
import cPickle
from collections import Counter
from ipdb import set_trace
from negative_samples import negative_sampler
import numpy as np
import os
from sma_toolkit import embeddings as emb_utils 
import streaming_pickle as stPickle
import time
np.set_printoptions(threshold=np.nan)
MIN_DOC_LEN=4

def get_parser():
    parser = argparse.ArgumentParser(description="Build Training Data")
    parser.add_argument('-input', type=str, required=True, help='train file(s)')
    parser.add_argument('-emb', type=str, required=True, help='path to word embeddings')
    parser.add_argument('-output', type=str, required=True, help='path of the output')
    parser.add_argument('-vocab_size', type=int, help='path of the output')
    parser.add_argument('-min_word_freq', type=int, help='ignore words that occur less than min_word_freq times',default=5)
    parser.add_argument('-seed', type=int, default=1234, help='random number generator seed')    
    parser.add_argument('-neg_samples', type=int, help='number of negative samples', default=10)
    return parser

if __name__ == "__main__" :
	parser = get_parser()
	args = parser.parse_args()	
	rng = np.random.RandomState(args.seed)    

	print "[input: %s | word vectors: %s | max vocab_size: %s | min_word_freq: %s | output: %s]" %  (os.path.basename(args.input), 
						 os.path.basename(args.emb), 
						 args.vocab_size, 
						 args.min_word_freq, 
						 os.path.basename(args.output)) 

	t0 = time.time()
	word_counter = Counter()
	n_docs=0	
	with open(args.input,"r") as fid:	
		for line in fid:			
			message = line.decode("utf-8").split()[1:]
			word_counter.update(message)				
			n_docs+=1
	#keep only words that occur at least min_word_freq times
	wc = {w:c for w,c in word_counter.items() if c>args.min_word_freq} 
	#keep only the args.vocab_size most frequent words
	tw = sorted(wc.items(), key=lambda x:x[1],reverse=True)
	top_words = {w[0]:i for i,w in enumerate(tw[:args.vocab_size])}	
	print "loading word embeddings..."		
	full_E, full_wrd2idx = emb_utils.read_embeddings(args.emb,top_words)
	ooevs = emb_utils.get_OOEVs(full_E, full_wrd2idx)
	#keep only words with pre-trained embeddings
	old_len = len(top_words)
	for w in ooevs:
		del top_words[w]	
	wrd2idx = {w:i for i,w in enumerate(top_words.keys())}	
	print "[vocabulary size: %d|%d]" % (len(wrd2idx),old_len)
	#generate the embedding matrix
	emb_size = full_E.shape[0]
	E = np.zeros((int(emb_size), len(wrd2idx)))   
	for wrd,idx in wrd2idx.items(): 
		E[:, idx] = full_E[:,top_words[wrd]]

	print "building training data..."
	#negative sampler
	idx2wrd = {i:w for w,i in wrd2idx.items()}	
	sampler = negative_sampler(word_counter, idx2wrd)

	if not os.path.exists(os.path.dirname(args.output)):
		os.makedirs(os.path.dirname(args.output))

	prev_user, prev_user_data, prev_ctxscores, prev_neg_samples  = None, [], [], []
	wrd_idx_counts = np.zeros(len(wrd2idx))	
	f_train = open(args.output,"wb") 
	
	with open(args.input,"r") as fid:		
		for j, line in enumerate(fid):		
			try:			
				message = line.replace("\"", "").replace("'","").split("\t")[1].decode("utf-8").split()	
			except:
				print "ignored line: {}".format(line)
			#convert to indices
			msg_idx = [wrd2idx[w] for w in message if w in wrd2idx]			
			#compute negative samples
			negative_samples = sampler.sample((len(msg_idx),args.neg_samples))			
			if len(msg_idx)<MIN_DOC_LEN: continue				
			u_idx = line.split("\t")[0] 								
			if prev_user is None: 
				#first time 
				prev_user = u_idx
			elif u_idx != prev_user:						
				#after accumulating all documents for current user, shuffle and write them to disk			
				assert len(prev_user_data) == len(prev_neg_samples)
				#shuffle the data			
				shuf_idx = np.arange(len(prev_user_data))
				rng.shuffle(shuf_idx)
				prev_user_data = [prev_user_data[i] for i in shuf_idx]
				prev_neg_samples = [prev_neg_samples[i] for i in shuf_idx]
				# set_trace()					
				split = int(len(prev_user_data)*.9)
				train = prev_user_data[:split]
				test  = prev_user_data[split:]	
				neg_samples = prev_neg_samples[:split]
				#each training instance consists of:
				#[user_name, train docs, test docs, negative samples] 			
				stPickle.s_dump_elt([prev_user, train, test, neg_samples ], f_train)				
				prev_user_data = []				
				prev_neg_samples = []							
			elif j == n_docs-1:			
				#can't forget the very last message
				prev_user_data.append(msg_idx)				
				prev_neg_samples.append(negative_samples)
				#shuffle the data			
				shuf_idx = np.arange(len(prev_user_data))
				rng.shuffle(shuf_idx)
				prev_user_data   = [prev_user_data[i] for i in shuf_idx]
				prev_neg_samples = [prev_neg_samples[i] for i in shuf_idx]		
				#split
				split = int(len(prev_user_data)*.9)				
				train = prev_user_data[:split]
				test  = prev_user_data[split:]				
				neg_samples =  prev_neg_samples[:split]
				stPickle.s_dump_elt([prev_user, train, test, neg_samples ], f_train)
				# print "  > user: %s (%d)" % (prev_user, len(train))
			prev_user = u_idx
			prev_user_data.append(msg_idx)
			prev_neg_samples.append(negative_samples)

			#collect word counts to compute unigram distribution
			for w_idx in msg_idx:								
				wrd_idx_counts[w_idx]+=1	
	f_train.close()					
	unigram_distribution = wrd_idx_counts / wrd_idx_counts.sum(0)	
	print "[pickling aux data]"
	#aux_data = os.path.split(args.output)[0] + "/aux.pkl"
	aux_data = os.path.splitext(args.output)[0] + "_aux.pkl"
	with open(aux_data,"wb") as fid:
		cPickle.dump([wrd2idx,unigram_distribution, word_counter, E], fid, cPickle.HIGHEST_PROTOCOL)
	tend = time.time() - t0
	print "\n[runtime: %d minutes (%.2f secs)]" % ((tend/60),tend)    