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
TRAIN_IDX = 1
NEG_SAMPLES_IDX = 4
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

def extract(instance, unigram_distribution, num_neg_samples):        
    train = instance[TRAIN_IDX]    
    neg_samples = []    
    for msg in train:            
        neg_samp = [ real_multinomial_samples(unigram_distribution, msg, num_neg_samples) \
                     for _ in xrange(len(msg)) ]
        neg_samples.append(neg_samp)        
    instance[NEG_SAMPLES_IDX] = neg_samples    
    return instance

def get_parser():
    parser = argparse.ArgumentParser(description="Extract negative samples")
    parser.add_argument('-input', type=str, required=True, help='train file')
    parser.add_argument('-aux_data', type=str, required=True, help='aux data file')
    parser.add_argument('-n_workers', type=int, help='number of jobs', default=1)    
    parser.add_argument('-negative_samples', type=int, help='number of negative samples', default=10)
    parser.add_argument('-resume', action="store_true", help='continue from a previous run', default=False)    
    return parser

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()      
    print "[training data: %s | aux_data: %s | n_workers: %d | resume: %s]" % (args.input, args.aux_data, args.n_workers, args.resume)             
    with open(args.aux_data,"r") as fid:
        wrd2idx, unigram_distribution, _ = cPickle.load(fid)        
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
                    if len(next_inst[NEG_SAMPLES_IDX]) == 0: current_instances.append(next_inst)
            else:
                for _ in xrange(args.n_workers): current_instances.append(training_data.next())
        except StopIteration:            
            done=True
        if args.n_workers>1:
            with Parallel(n_jobs=args.n_workers) as parallel:
                res = parallel(delayed(extract)(instance, unigram_distribution, args.negative_samples) for instance in current_instances)            
        else:            
            res = [extract(instance,unigram_distribution, args.negative_samples) for instance in current_instances]    
        # print res
        for r in res: stPickle.s_dump_elt(r,new_training_data)       
        j+=len(current_instances)
        t_i = (time.time() - t0) / 60 
        sys.stdout.write("\r>:%d (~%d mins)" % (j,t_i))
        sys.stdout.flush()     
    #replace the training data file with the new augmented one    
    os.remove(args.input)
    os.rename(tmp_data_path, args.input)
    tend = time.time() - t0
    print "[runtime: %d minutes (%.2f secs)]" % ((tend/60),tend)    
