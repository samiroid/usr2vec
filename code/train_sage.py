import cPickle
from collections import Counter
# from ipdb import set_trace
import numpy as np
from my_utils import colstr
import time
import sys
import streaming_pickle as stPickle
from SAGE import SAGE
from joblib import Parallel, delayed
import warnings
np.set_printoptions(threshold=np.nan)
warnings.filterwarnings("ignore")

def stream_estimate_user(instance):	
	user,train_matrix,test_matrix = instance	
	obj=0
	user_ll=0		
	n_batches = (train_matrix.shape[1]*1./mbsize) 
	#if the number of batches is not a whole number add one
	if n_batches%1 != 0: n_batches+=1    
 	n_batches = int(n_batches)
	for j in xrange(n_batches):		
		try:
			d = train_matrix[:,j*mbsize:(j+1)*mbsize].todense()
		except IndexError:			
			d = train_matrix[:,j*mbsize:train_matrix.shape[1]].todense()
		obj += sage.train(user,d)			
	#evaluate 			
	user_ll = sage.evaluate(user, test_matrix.todense())
	# set_trace()
	return obj/train_matrix.shape[1], user_ll/test_matrix.shape[1], user, sage.user_etas.get_value()[:,user]				 

if __name__ == "__main__":
	
	aux_pickle, train_data_path, sage_params_path, n_jobs = sys.argv[1:]
	n_jobs = int(n_jobs)
	print "Loading data"
	with open(aux_pickle,"r") as fid:
		wrd2idx,usr2idx,back_word_probs,_,_ = cPickle.load(fid)	
		
	t0 = time.time()	
	#training parameters 
	#lrate = 0.00005 + epochs = 200 + mbsize = 400 --> .615
	lrate = 0.00005	
	epochs = 200
	mbsize = 400
	patience = 5	
	sage = SAGE(back_word_probs, len(usr2idx), initial_etas=None, lrate=lrate)
	current_sage_params = sage.user_etas.get_value()
	prev_ll = -10000
	prev_obj = 0
	best_obj = -1000000
	drops = 0		
	tf = open(train_data_path,"rb")		
	best_ll = 0
	print "training"
	print "mbsize: %d | lrate: %.5f" % (mbsize,lrate)
	for e in xrange(epochs):	
		obj=0	
		user_ll = 0 
		t0_in = time.time()
		done = False
		n_examples = 0
		training_data = stPickle.s_load(tf)        
		while not done:
			try:
				current_samples = []
				for _ in xrange(n_jobs): current_samples.append(training_data.next())
			except StopIteration:  				
				done=True			
			res = [ stream_estimate_user(instance) for instance in current_samples]
			# res = Parallel(n_jobs=n_jobs)(delayed(stream_estimate_user)(instance)
	  #                              for instance in current_samples)
			for r in res:
				obj    += r[0]
				user_ll+= r[1]
				user    = r[2]
				params  = r[3]  
				#update user parameters with the result of this training epoch
				current_sage_params[:,user] = params
			n_examples+=len(res)
		#average objective and user likelihood
		obj/=n_examples
		user_ll/=n_examples
		#update model with new user parameters
		sage.user_etas.set_value(current_sage_params)
		color_obj=None
		if obj > prev_obj:
			color_obj="green"
			if obj>best_obj:
				best_obj=obj
		elif obj < prev_obj:
			color_obj="red"
		prev_obj=obj
		z = colstr(("%.2f " % obj), color_obj, best_obj==obj)
		# set_trace() 
		sys.stdout.write("\repoch:%d\%d | obj: " % (e,epochs) +z) 
		sys.stdout.flush()
		# set_trace()
		t1_in = time.time() - t0_in
		elapsed = time.time() - t0
		# print test_size
		color = None		
		if user_ll > prev_ll:
			color='green'
			drops = 0
			if user_ll > best_ll:
				#keep the best params								
				with open(sage_params_path,"w") as fod:
					cPickle.dump([sage.user_etas.get_value(),back_word_probs], fod,cPickle.HIGHEST_PROTOCOL)
				best_ll = user_ll										
		elif user_ll < prev_ll:
			color='red'
			drops+=1			
		else:
			drops+=1		
		print " likelihood: " + colstr(("%.3f" %user_ll),color,(user_ll==best_ll)) +  " (%.2f\%.2f secs)" % (t1_in,elapsed)		
		
		if drops>=patience:
			print "Ran out of patience...I'm out!"
			break	
		prev_ll = user_ll
		#reset 
		tf.seek(0)
t1 = time.time() - t0
print "took: %d minutes" % (t1/60)