import argparse
import cPickle 
from my_utils import colstr
from pdb import set_trace
import usr2vec
import sys
import warnings
import time
import os
import random
import streaming_pickle as stPickle
warnings.filterwarnings("ignore")
import numpy as np

def count_users(dataset):	
	with open(dataset) as fid:
		return len([z for z in stPickle.s_load(fid)])

def get_parser():
    parser = argparse.ArgumentParser(description="Train U2V")
    parser.add_argument('-input', type=str, required=True, help='train file(s)')
    parser.add_argument('-aux', type=str, required=True, help='aux data file')
    parser.add_argument('-output', type=str, help='output path')
    parser.add_argument('-lrate', type=float, default=5e-5, help='learning rate')
    parser.add_argument('-margin', type=int, default=1, help='margin size')
    parser.add_argument('-epochs', type=int, default=25, help='number of training epochs')
    parser.add_argument('-patience', type=int, default=5, help='stop training if no progress made after this number of epochs')
    parser.add_argument('-reshuff', default=False, action="store_true", help='if True, instances will be reshuffled after each training epoch')
    parser.add_argument('-init_w2v', choices=["mean",'gauss'], help='initialize user embeddings with information from the word embeddings')

    return parser

if __name__ == "__main__":
	#command line arguments	
	parser = get_parser()
	args = parser.parse_args()	
	print "loading data..."	
	with open(args.aux,"r") as fid:
		_,_,E = cPickle.load(fid) 
	try:
		n_usrs = count_users(args.input)
	except IOError:
		print "Couldn't not find file %s" % args.input
		sys.exit()	
	#path to save intermediate versions of the user embedding matrix (during training)
	tmp_name = ''.join([ chr(random.randint(97,122)) for i in xrange(10)])
	user_emb_bin = os.path.split(args.input)[0]+"/tmp-"+tmp_name.upper() 
	print "[lrate: %.5f | margin loss: %d | epochs: %d| reshuff: %s | init_w2v: %s | @%s]\n" % (args.lrate,args.margin, args.epochs, args.reshuff, args.init_w2v, user_emb_bin)	
	if args.init_w2v:
		u2v = usr2vec.Usr2Vec(E, n_usrs,lrate=args.lrate,margin_loss=args.margin, init_w2v=args.init_w2v)	
	else:
		u2v = usr2vec.Usr2Vec(E, n_usrs,lrate=args.lrate,margin_loss=args.margin)	
	t0 = time.time()				
	prev_logprob, best_logprob = -10**100, -10**100	 
	prev_obj, best_obj  = 10**100, 10**100
	drops = 0
	usr2idx = {}
	tf = open(args.input,"r")	
	for e in xrange(args.epochs):	
		obj      = 0
		log_prob = 0
		ts = time.time()						
		############# TRAIN 	
		#rewind
		tf.seek(0)		
		training_data = stPickle.s_load(tf)   	
		for instance in training_data:			
			user, train, test, cond_probs, neg_samples = instance
			try:
				u_idx  = usr2idx[user]
			except KeyError:
				u_idx = len(usr2idx)
				usr2idx[user] = u_idx				
			sys.stdout.write("\rtraining: %s  " % user)
			sys.stdout.flush()	
			if args.reshuff:
				for x in np.random.permutation(len(train)):
					obj += u2v.train(u_idx, train[x], neg_samples[x], cond_probs[x])		
			else:
				for msg_train, neg, cp in zip(train,neg_samples,cond_probs): 				
					obj += u2v.train(u_idx, msg_train, neg, cp)		
		#average objective 
		obj/=len(usr2idx)			
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

		############# EVALUATE 
		#rewind
		tf.seek(0)	
		training_data = stPickle.s_load(tf)   				
		for instance in training_data:
			user, train, test, cond_probs, neg_samples = instance				
			u_idx  = usr2idx[user]
			user_logprob=0		
			for msg_test in test: 			
				l,all_prob = u2v.predict(u_idx, msg_test)
				user_logprob+= l
			log_prob += (user_logprob/len(test))
		log_prob/=len(usr2idx)	
		color=None		
		if log_prob > prev_logprob:				
			color='green'				
			if log_prob > best_logprob:	
				#keep best model			
				u2v.save_model(user_emb_bin)			
				best_logprob=log_prob			
		elif log_prob < prev_logprob: 
			drops+=1
			color='red'
		else:
			drops+=1					 		
		print " user inv log prob: " + colstr(("%.3f" % log_prob), color, (best_logprob==log_prob))  
		if drops>=args.patience:
			print "ran out of patience..."
			break
		prev_logprob = log_prob

	print "[runtime: ~%d minutes]" % ((time.time()-t0)/60)	
	tf.close()	

	############# EXPORT

	print "Exporting embeddings..."
	with open(user_emb_bin,"r") as fid:
		U = cPickle.load(fid)[0]
	with open(args.output,"w") as fod:
		fod.write("%d %d\n" % (U.shape[1],U.shape[0]))	
		for user, u_id in usr2idx.items():		
			emb = U[:,u_id]
			fod.write("%s %s\n" % (user, " ".join(map(str, emb))))
		
