import cPickle
import gensim
from joblib import Parallel, delayed
# from ipdb import set_trace
import numpy as np
import os
import SAGE
import streaming_pickle as stPickle
import sys
import time

SIZE  = 400 
NEG_SAMP_SIZE = 10
WINDOW_SIZE = 3
rng = np.random.RandomState(1234)
PAD_TOKEN = u'_PAD_'

def negative_samples(unigram_distribution, exclude=[], n_samples=1):    
        samples = []        
        while len(samples) != n_samples:
            vals = np.random.multinomial(1, unigram_distribution)        
            wrd_idx = np.nonzero(vals)[0][0]
            if wrd_idx not in exclude: samples.append(wrd_idx)
        return samples

def parallel_extract(i, instance, unigram_distribution):        
    train = instance[1]    
    neg_samples = []
    sys.stdout.write("\ri:%d " % i)
    sys.stdout.flush()    
    for msg in train:            
        neg_samp = [negative_samples(unigram_distribution,msg,NEG_SAMP_SIZE) for _ in xrange(len(msg))]        
        neg_samples.append(neg_samp)    
    
    instance[4] = neg_samples
    
    return instance
    
if __name__ == "__main__":

    #command line arguments
    stuff_pickle, training_data_path, n_jobs = sys.argv[1:]
    
    n_jobs = int(n_jobs)
    print "Loading data"    
    with open(stuff_pickle,"r") as fid:
        wrd2idx,usr2idx,unigram_distribution,_ = cPickle.load(fid)    
    
    t0 = time.time()
    prev_time = time.time()    
    print "Computing negative samples"    
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
            res = Parallel(n_jobs=n_jobs)(delayed(parallel_extract)(i, instance,unigram_distribution) 
                                      for i, instance in enumerate(current_instances))
        else:
            res = [parallel_extract(i, instance,unigram_distribution) for i,instance in enumerate(current_instances)]    

        for r in res: stPickle.s_dump_elt(r,new_training_data)        
            
    tend = time.time() - t0
    print "It took: %d minutes (%.2f secs)" % ((tend/60),tend)    
    #replace the training data file with the new augmented one    
    os.remove(training_data_path)
    os.rename(new_training_data_path, training_data_path)
