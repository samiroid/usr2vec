import argparse
import cPickle
from joblib import Parallel, delayed
# from ipdb import set_trace
import numpy as np
import os
import streaming_pickle as stPickle
import sys
import time

# def get_samples(unigram_distribution, exclude=[], n_samples=1):    
#         samples = []        
#         while len(samples) != n_samples:
#             vals = np.random.multinomial(1, unigram_distribution)        
#             wrd_idx = np.nonzero(vals)[0][0]
#             if wrd_idx not in exclude: samples.append(wrd_idx)        
#         return samples

def multinomial_samples(unigram_distribution, exclude=[], n_samples=1):    
        samples = []        
        while len(samples) != n_samples:
            vals = np.random.multinomial(1, unigram_distribution)        
            wrd_idx = np.nonzero(vals)[0][0]            
            if wrd_idx not in exclude: samples.append(wrd_idx)        
        return samples

def real_multinomial_samples(unigram_distribution, exclude=[], n_samples=1):    
        samples = []        
        while len(samples) != n_samples:            
            wrd_idx = np.argmax(np.random.multinomial(1, unigram_distribution))
            # from ipdb import set_trace; set_trace()
            if wrd_idx not in exclude: 
                samples.append(wrd_idx)                        
        return samples

def random_samples(exclude=[], n_samples=1):
    pass

def fast_multinomial_samples(unigram_distribution, exclude=[], n_samples=1):  
        #generate more negative samples than needed then remove words from exclude list          
        candidates = np.argmax(np.random.multinomial(n_samples,unigram_distribution,n_samples*2),axis=1)       
        return list(set(candidates)-set(exclude))[:n_samples]

def extract(i, instance, unigram_distribution, num_neg_samples):        
    train = instance[1]    
    neg_samples = []
    sys.stdout.write("\ri:%d " % i)
    sys.stdout.flush()    
    for msg in train:            
        neg_samp = [ real_multinomial_samples(unigram_distribution, msg, num_neg_samples) \
                     for _ in xrange(len(msg)) ]
        neg_samples.append(neg_samp)        
    instance[4] = neg_samples    
    return instance

def get_parser():
    parser = argparse.ArgumentParser(description="Extract negative samples")
    parser.add_argument('-input', type=str, required=True, help='train file')
    parser.add_argument('-aux_data', type=str, required=True, help='aux data file')
    parser.add_argument('-n_workers', type=int, help='number of jobs', default=1)    
    parser.add_argument('-negative_samples', type=int, help='number of negative samples', default=10)
    return parser

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()      
    print "[training data: %s | aux_data: %s | n_workers: %d]" % (args.input, args.aux_data, args.n_workers)             
    with open(args.aux_data,"r") as fid:
        wrd2idx, unigram_distribution, _ = cPickle.load(fid)        
    t0 = time.time()
    prev_time = time.time()    
    print "extracting negative samples..."    
    tdf = open(args.input,"r")
    training_data = stPickle.s_load(tdf)        
    tmp_data_path = args.input.strip(".pkl")+"_new.pkl"
    new_training_data = open(tmp_data_path,"w")    
    done=False
    while not done:        
        try:
            current_instances = []
            for _ in xrange(args.n_workers): current_instances.append(training_data.next())
        except StopIteration:            
            done=True
        if args.n_workers>1:
            with Parallel(n_jobs=args.n_workers) as parallel:
                res = parallel(delayed(extract)(i, instance, unigram_distribution, args.negative_samples) for i, instance in enumerate(current_instances))
            # res = Parallel(n_jobs=args.n_workers)(delayed(extract)(i, instance, unigram_distribution, args.negative_samples) for i, instance in enumerate(current_instances))
        else:            
            res = [extract(i, instance,unigram_distribution, args.negative_samples) for i,instance in enumerate(current_instances)]    
        # print res
        for r in res: stPickle.s_dump_elt(r,new_training_data)        
    #replace the training data file with the new augmented one    
    os.remove(args.input)
    os.rename(tmp_data_path, args.input)
    tend = time.time() - t0
    print "[runtime: %d minutes (%.2f secs)]" % ((tend/60),tend)    
