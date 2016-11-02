import cPickle
import gensim
from joblib import Parallel, delayed
# from ipdb import set_trace
import numpy as np
import os
import streaming_pickle as stPickle
import sys
import time

WINDOW_SIZE = 3
rng = np.random.RandomState(1234)
PAD_TOKEN = u'_PAD_'

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

def parallel_extract(i, instance):        
    train = instance[1]
    cond_probs  = []    
    sys.stdout.write("\ri:%d " % i)
    sys.stdout.flush()    
    for msg in train:
        #convert word indices back to words
        actual_tokens = [idx2wrd[idx] for idx in msg]
        # ##### CONDITIONAL PROBABILITIES OF WORDS GIVEN CONTEX FOR ALL THE WORDS
        cp = logprob_words_context(w2v, actual_tokens, WINDOW_SIZE)
        # from ipdb import set_trace; set_trace()
        cond_probs.append(np.array(cp,dtype='float32'))
    instance[3] = cond_probs
    return instance
    
if __name__ == "__main__":
    #command line arguments
    training_data_path, aux_data_path, embs_path, n_jobs  = sys.argv[1:]
    n_jobs = int(n_jobs)
    print "[training data: %s]" % training_data_path 
    print "[aux data: %s]" % aux_data_path
    print "Loading..."
    w2v = gensim.models.Word2Vec.load(embs_path)
    with open(aux_data_path,"r") as fid:        
        wrd2idx,_,_ = cPickle.load(fid)        
    #index 2 actual word
    idx2wrd = {v:k for k,v in wrd2idx.items()}        
    t0 = time.time()
    prev_time = time.time()    
    print "Computing conditional word probabilities"    
    tdf = open(training_data_path,"r")
    training_data = stPickle.s_load(tdf)        
    new_training_data_path = training_data_path.strip(".pkl")+"_new.pkl"
    new_training_data = open(new_training_data_path,"w")    
    done=False
    while not done:
        try:
            current_instances = []
            for _ in xrange(n_jobs): current_instances.append(training_data.next())
        except StopIteration:            
            done=True
        if n_jobs>1:
            res = Parallel(n_jobs=n_jobs)(delayed(parallel_extract)(i, instance)  for i, instance in enumerate(current_instances))
        else:
            res = [parallel_extract(i, instance) for i,instance in enumerate(current_instances)]    
        for r in res: stPickle.s_dump_elt(r,new_training_data)        
    tend = time.time() - t0    
    #replace the training data file with the new augmented one    
    os.remove(training_data_path)
    os.rename(new_training_data_path, training_data_path)
    print "It took: %d minutes (%.2f secs)" % ((tend/60),tend)    