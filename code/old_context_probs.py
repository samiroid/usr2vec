import argparse
import cPickle
import gensim
from joblib import Parallel, delayed
# from ipdb import set_trace
import numpy as np
import os
import streaming_pickle as stPickle
import sys
import time

# WINDOW_SIZE = 3
rng = np.random.RandomState(1234)
PAD_TOKEN = u'_pad_'
CTX_PROBS_IDX = 3
TRAIN_IDX = 1

def logprob_words_context(w2v, tokens, window_size): 
    """
        Given a sentence of tokens = {w1, ..., wn} computes
        logP(w_a|w_b) for all windows of size n
        the message is padded to ensure that all windows are equal
    """
    padd = [PAD_TOKEN] * window_size
    padded_m = padd + tokens + padd 
    #vector with the log probs in the same order of the words in the message    
    cp_array = [0]*len(tokens)
    for i in xrange(len(tokens)):        
        #left window
        wl = padded_m[i:i+window_size]
        #right window
        wr = padded_m[i+window_size+1:(i+1)+window_size*2]
        context_window = wl + wr
        center_word = padded_m[i+window_size]           
        #don't repeat computations for the same pair of words
        word_pairs =  [ [center_word,ctx_word] for 
                        ctx_word in set(context_window) ]                  
        word_scores = w2v.score(word_pairs)
        cp_array[i] = round(sum(word_scores),3)        
    return cp_array

def parallel_extract(instance, window_size):        
    train = instance[TRAIN_IDX]
    cond_probs  = []        
    for msg in train:
        #convert word indices back to words
        actual_tokens = [idx2wrd[idx] for idx in msg]
        # ##### CONDITIONAL PROBABILITIES OF WORDS GIVEN CONTEX FOR ALL THE WORDS
        cp = logprob_words_context(w2v, actual_tokens, window_size)
        # from ipdb import set_trace; set_trace()
        cond_probs.append(np.array(cp,dtype='float32'))
    instance[CTX_PROBS_IDX] = cond_probs
    return instance

def get_parser():
    parser = argparse.ArgumentParser(description="Compute context log probabilities for each window of each document")
    parser.add_argument('-input', type=str, required=True, help='train file')
    parser.add_argument('-emb', type=str, required=True, help='path to word embeddings')
    parser.add_argument('-aux_data', type=str, required=True, help='aux data file')
    parser.add_argument('-n_workers', type=int, help='number of jobs', default=1)    
    parser.add_argument('-resume', action="store_true", help='continue from a previous run', default=False)    
    parser.add_argument('-window_size', type=int, default=3, help='window size')    
    return parser

if __name__ == "__main__":
    #command line arguments    
    parser = get_parser()
    args = parser.parse_args()      
    print "[training data: %s | aux_data: %s | n_workers: %d | window: %d | resume: %s]" % (args.input, args.aux_data, args.n_workers, args.window_size, args.resume)     
    w2v = gensim.models.Word2Vec.load(args.emb)
    with open(args.aux_data,"r") as fid:        
        wrd2idx,_,_ = cPickle.load(fid)        
    #index 2 actual word
    idx2wrd = {v:k for k,v in wrd2idx.items()}        
    t0 = time.time()
    prev_time = time.time()        
    tdf = open(args.input,"r")
    training_data = stPickle.s_load(tdf)        
    tmp_data_path = args.input.strip(".pkl")+"_new.pkl"
    new_training_data = open(tmp_data_path,"w")    
    done=False
    j=0 #number of instances processed
    while not done:
        try:
            current_instances = []
            if args.resume:
                while len(current_instances) < args.n_workers: 
                    next_inst = training_data.next()
                    #ignore the instances where these quantities have already been calculated
                    if len(next_inst[CTX_PROBS_IDX]) == 0:
                        current_instances.append(next_inst)
            else:
                for _ in xrange(args.n_workers): current_instances.append(training_data.next())
        except StopIteration:            
            done=True
        if args.n_workers>1:            
            res = Parallel(n_jobs=args.n_workers)(delayed(parallel_extract)(instance, args.window_size)  for instance in current_instances)            
        else:
            res = [parallel_extract(instance, args.window_size) for instance in current_instances]   
        for r in res: stPickle.s_dump_elt(r,new_training_data)  
        j+=len(current_instances)
        t_i = (time.time() - t0) / 60 
        sys.stdout.write("\r>:%d (~%d mins)" % (j,t_i))
        sys.stdout.flush()          
    tend = time.time() - t0    
    #replace the training data file with the new augmented one    
    os.remove(args.input)
    os.rename(tmp_data_path, args.input)
    print "[runtime: %d minutes (%.2f secs)]" % ((tend/60),tend)    