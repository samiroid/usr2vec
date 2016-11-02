import cPickle 
from my_utils import colstr
from pdb import set_trace
import usr2vec
import sys
import warnings
import time
import streaming_pickle as stPickle
warnings.filterwarnings("ignore")
import numpy as np

if __name__ == "__main__":
	#command line arguments
	train_path, usr2idx_path, aux_data_path, user_embs_bin_path, user_embs_txt_path  = sys.argv[1:]
	print "Loading data"	
	with open(aux_data_path,"r") as fid:
		wrd2idx,_,E = cPickle.load(fid)    		
	with open(usr2idx_path,"r") as fid:
		usr2idx = cPickle.load(fid)    		
	n_usrs = len(usr2idx)
	#model hyperparams
	# lrate  = 0.00001
	lrate  = 0.0001
	m      = 1 
	epochs = 25	
	#epochs = 1	
	patience = 5
	drops    = 0
	u2v = usr2vec.Usr2Vec(E, n_usrs,lrate=lrate,margin_loss=m)	
	t0 = time.time()	
	print "training: lrate: %.5f | margin loss: %d | epochs: %d\n" % (lrate,m,epochs)
	tf = open(train_path,"r")	
	prev_logprob = -10000	
	best_logprob = -10000	
	prev_obj     = 100000
	best_obj     = 100000
	for e in xrange(epochs):	
		obj = 0
		log_prob = 0
		ts = time.time()
		done = False
		n_instances = 0
		training_data = stPickle.s_load(tf)   
		#train				
		for instance in training_data:			
			user, train, test, cond_probs, neg_samples = instance
			n_instances+=1			
			sys.stdout.write("\rtraining: %d  " % n_instances)
			sys.stdout.flush()			
			for msg_train, neg, cp in zip(train,neg_samples,cond_probs): 
				obj += u2v.train(user, msg_train, neg, cp)			
		#average objective 
		obj/=n_instances			
		obj_color = None		
		if obj < prev_obj:
			obj_color='green'
			if obj < best_obj:				
				best_obj=obj
		elif obj > prev_obj:
			color='red'
		prev_obj=obj		
		et = time.time() - ts
		obj_str = colstr(("%.3f" % obj),obj_color,(best_obj==obj))
		sys.stdout.write("\rEpoch:%d | obj: " % (e+1) + obj_str +" | " +  " (%.2f secs)" % (et) ) 
		sys.stdout.flush()	
		if not e%5: #evaluate each 5 epochs			
			#rewind
			tf.seek(0)	
			training_data = stPickle.s_load(tf)   		
			for instance in training_data:
				user, train, test, cond_probs, neg_samples = instance
				user_logprob=0		
				for msg_test in test: 			
					l,all_prob = u2v.predict(user, msg_test)
					user_logprob+= l
				log_prob += (user_logprob/len(test))

			log_prob/=n_instances
			color=None		
			if log_prob > prev_logprob:
				drops=0
				color='green'				
				if log_prob > best_logprob:	
					#keep best model			
					u2v.save_model(user_embs_bin_path)				
					best_logprob=log_prob
			elif log_prob < prev_logprob: 
				drops+=1
				color='red'
			else:
				drops+=1		
			print " user inv log prob: " + colstr(("%.3f" % log_prob), color, (best_logprob==log_prob))  + "~%.3f" % np.exp(log_prob) 

			if drops>=patience:
				print "ran out of patience..."
				break
			prev_logprob = log_prob
		else:
			print ""	
		#rewind
		tf.seek(0)
		
	print "Took: %d minutes" % ((time.time()-t0)/60)	
	tf.close()	
	print "Exporting embeddings..."
	with open(user_embs_bin_path,"r") as fid:
		U = cPickle.load(fid)[0]
	with open(user_embs_txt_path,"w") as fod:
		fod.write("%d %d\n" % (U.shape[1],U.shape[0]))	
		for user, u_id in usr2idx.items():		
			emb = U[:,u_id]
			fod.write("%s %s\n" % (user, " ".join(map(str, emb))))
		
