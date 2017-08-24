import argparse
import cPickle 
from sma_toolkit import colstr
#from ipdb import set_trace
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
    parser.add_argument('-quiet', default=False, action="store_true", help='do not print training objectives and validation error')

    return parser

if __name__ == "__main__":
	#command line arguments	
	parser = get_parser()
	args = parser.parse_args()	
	print "[Training U2V by user]"
	print "loading data..."	
	with open(args.aux,"r") as fid:
		_,_,_,E = cPickle.load(fid) 
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
	total_time = time.time()					
	usr2idx = {}
	tf = open(args.input,"r")	
	
	training_data = stPickle.s_load(tf)   	
	#each training instance corresponds to a user
	total_logprob = 0  
	total_epochs = 0
	for z, instance in enumerate(training_data):	
		# if z == 100:
		# 	print "bailed earlier with %d users " % z
		# 	n_usrs = z
		# 	break
		prev_logprob, best_logprob = -10**100, -10**100	 
		prev_obj, best_obj  = 10**100, 10**100
		drops = 0		
		curr_lrate = u2v.lrate
		user, train, test, neg_samples = instance
		try:
			u_idx  = usr2idx[user]
		except KeyError:
			u_idx = len(usr2idx)
			usr2idx[user] = u_idx		
		if not args.quiet:		
			print "[user: %s (%d/%d)]" % (user,z+1,n_usrs)		
		user_time  = time.time()	
		for e in xrange(args.epochs):	
			############# TRAIN 	
			total_epochs+=1
			obj = 0					
			prev_lrate = curr_lrate
			if args.reshuff:
				for x in np.random.permutation(len(train)):
					obj += u2v.train(u_idx, train[x], neg_samples[x], curr_lrate)
			else:
				for msg_train, neg in zip(train, neg_samples): 				
					obj += u2v.train(u_idx, msg_train, neg, curr_lrate)			
			#average objective 
			obj/=len(train)			
			obj_color = None		
			if obj < prev_obj:
				obj_color='green'
				if obj < best_obj:				
					best_obj=obj
			elif obj > prev_obj:
				color='red'
			prev_obj=obj	
			if not args.quiet:					
				obj_str = colstr(("%.3f" % obj),obj_color,(best_obj==obj))
				sys.stdout.write("\r\tepoch:%d | obj: %s" % ((e+1),obj_str)) 	
				sys.stdout.flush()			

			############# EVALUATE 
			logprob=0		
			for msg_test in test: 			
				l,all_prob = u2v.predict(u_idx, msg_test)			
				logprob+= l
			logprob/=len(test)	
			logprob=round(logprob,4)			
			color=None		
			if logprob > prev_logprob:				
				color='green'				
				if logprob > best_logprob:	
					#keep best model			
					u2v.save_model(user_emb_bin)			
					best_logprob=logprob			
			elif logprob < prev_logprob: 
				drops+=1
				color='red'
				#decay the learning rate exponentially
				curr_lrate*=10**-1
				#curr_lrate/=2
			else:
				drops+=1
			if not args.quiet:	
				if curr_lrate!=prev_lrate:
					print " ILL: " + colstr(("%.3f" % logprob), color, (best_logprob==logprob)) + " (lrate:" + str(curr_lrate)+")" 
				else:
					print " ILL: " + colstr(("%.3f" % logprob), color, (best_logprob==logprob))			
			if drops>=args.patience:
				print "ran out of patience (%d epochs)" % e
				break
			prev_logprob = logprob
		et = time.time() - user_time
		user_mins = np.floor(et*1.0/60)
		user_secs = et - user_mins*60
		tt = time.time() - total_time
		tt_mins = np.floor(tt*1.0/60)
		tt_secs = tt - tt_mins*60
		total_logprob+=best_logprob
		alp = total_logprob/(z+1)
		if not args.quiet:
			print "> ILL: %.3f %d.%d mins (avg ILL: %.3f| %d.%d mins)" % (best_logprob,user_mins,user_secs,alp,tt_mins,tt_secs)
		
	tt = time.time() - total_time
	mins = np.floor(tt*1.0/60)
	secs = tt - mins*60	
	print "*"*90
	print "[avg epochs: %d | avg ILL: %.4f | lrate: %.5f]" % ((total_epochs/n_usrs),(total_logprob/n_usrs),args.lrate)
	print "*"*90
	print "[runtime: %d.%d minutes]" % (mins,secs)	
	tf.close()	

	############# EXPORT

	print "Exporting embeddings..."
	with open(user_emb_bin,"r") as fid:
		U = cPickle.load(fid)[0]
	#create dir if it does not exist
	if not os.path.exists(os.path.dirname(args.output)):
		os.makedirs(os.path.dirname(args.output))
	with open(args.output+".txt","w") as fod:
		fod.write("%d %d\n" % (U.shape[1],U.shape[0]))	
		for user, u_id in usr2idx.items():		
			emb = U[:,u_id]
			fod.write("%s %s\n" % (user, " ".join(map(str, emb))))
		
