import cPickle
from collections import Counter
from ipdb import set_trace
import numpy as np
import operator
import sys
import streaming_pickle as stPickle
from scipy.sparse import csc_matrix

rng = np.random.RandomState(1234)     
np.set_printoptions(threshold=np.nan)

def as_sparse_matrix(indices,vocab_size):	
	rows = []
	cols = []
	data = []
	for i,doc in enumerate(indices):			
		for w,c in Counter(doc).items(): 
			rows.append(w)
			cols.append(i)
			data.append(c)			
	return csc_matrix((data, (rows, cols)), shape=(vocab_size,len(indices)),dtype='int32')

def as_dense_matrix(indices,vocab_size):		
	matrix = np.zeros((vocab_size,len(indices))).astype('int32')
	for i,doc in enumerate(indices):			
		for w,c in Counter(doc).items(): matrix[w,i] = c
	return matrix

tweets, emb_path, training_data_path, aux_data, usr2idx_path, n_slices = sys.argv[1:]

#if "clean" not in tweets:
#	raise EnvironmentError("Requires \"clean\" file...")

MAX_WORDS = 50000
word_counter = Counter()
n_docs=0
print "building vocabulary..."
all_users = []
with open(tweets,"r") as fid:	
	for line in fid:	
		usr = line.split("\t")[0] 			
		all_users += [usr]
		message = line.split("\t")[1].decode("utf-8").split()
		word_counter.update(message)				
		n_docs+=1
#keep only the MAX_WORDS more frequent words
sw = sorted(word_counter.items(), key=lambda x:x[1],reverse=True)
top_words = {w[0]:None for w in sw[:MAX_WORDS]}
print "extracting pre-trained word embeddings..."
with open(emb_path) as fid:        
    # Get emb size
    _, emb_size = fid.readline().split()            
    z=0
    for line in fid.readlines():      	
    	# z+=1
    	# if z>=10:
    	# 	print "out early!!!!!!"
    	# 	break
        items = line.split()
        wrd   = items[0]
        e = np.array(items[1:]).astype(float)            
        if wrd in top_words:
        	top_words[wrd] = e        
#generate the embedding matrix
#keep only words with pre-trained embedding
words_with_embedding = [i for i,j in top_words.items() if j is not None]
wrd2idx = {w:i for i,w in enumerate(words_with_embedding)}
E = np.zeros((int(emb_size), len(wrd2idx)))   
for wrd,idx in wrd2idx.items(): E[:, idx] = top_words[wrd]
print "building training data..."
n_users = len(set(all_users))
n_slices = int(n_slices)
print "[unique users: %d | #slices: %d]" % (n_users, n_slices)
users_per_slice = n_users*1./n_slices
if (users_per_slice - int(users_per_slice)) > 0.5: users_per_slice = int(users_per_slice) + 1
print "[users/slice: %d]" % users_per_slice
prev_user, prev_user_data = 0, []
word_counts = np.zeros(len(wrd2idx))
f_train       = open(training_data_path,"wb") 
f_usr2idx     = open(usr2idx_path,"wb") 
#keep track of which files were created
train_files   = [training_data_path[training_data_path.rfind("/")+1:]]
usr2idx_files = [usr2idx_path[usr2idx_path.rfind("/")+1:]]
file_counter=1
usr2idx={}
with open(tweets,"r") as fid:		
	for j, line in enumerate(fid):					
		message = line.split("\t")[1].decode("utf-8").split()	
		#convert to indices
		msg_idx = [wrd2idx[w] for w in message if w in wrd2idx]
		#TODO: I might need to remove this as it might influence later logic
		if len(msg_idx)==0: continue					
		user = line.split("\t")[0] 			
		try:			
			u_idx = usr2idx[user] 
		except KeyError:
			#if not found use the next sequential number
			usr2idx[user] = len(usr2idx)
			u_idx = usr2idx[user]		
		#after accumulating all documents for current user, shuffle and write them to disk		
		if u_idx != prev_user:			
			#shuffle the data					
			rng.shuffle(prev_user_data)
			split = int(len(prev_user_data)*.9)
			train = prev_user_data[:split]
			test  = prev_user_data[split:]				
			stPickle.s_dump_elt([prev_user, train, test, [],[]], f_train)
			prev_user_data = []				
			#partioning the training data
			if len(usr2idx)>int(users_per_slice):
				#the last user will be part of the next partition
				last_user = max(usr2idx.iteritems(), key=operator.itemgetter(1))[0]								
				del usr2idx[last_user]
				#write usr2idx and reset
				cPickle.dump(usr2idx, f_usr2idx)	
				u_idx = 0			
				usr2idx = {last_user:u_idx}
				#close current files
				f_train.close()
				f_usr2idx.close()
				#open next files
				file_counter += 1
				next_training_file = training_data_path.replace(".pkl",str(file_counter))+".pkl"			
				next_usr2idx_file  = usr2idx_path.replace(".pkl",str(file_counter))+".pkl"
				f_train   = open(next_training_file,"wb") 
				f_usr2idx = open(next_usr2idx_file,"wb") 				
				#keep track of which files were created
				train_files   += [next_training_file[next_training_file.rfind("/")+1:]]				
				usr2idx_files += [next_usr2idx_file[next_usr2idx_file.rfind("/")+1:]]
		elif j == n_docs-1:	
			#take into account the very last message	
			prev_user_data.append(msg_idx)
			#shuffle the data		
			rng.shuffle(prev_user_data)
			split = int(len(prev_user_data)*.9)				
			train = prev_user_data[:split]
			test  = prev_user_data[split:]
			stPickle.s_dump_elt([prev_user, train, test, [],[]], f_train)
			#write usr2idx	
			cPickle.dump(usr2idx, f_usr2idx)				
			#close current files
			f_train.close()
			f_usr2idx.close()
		prev_user = u_idx
		prev_user_data.append(msg_idx)
		#collect word counts to compute unigram distribution
		for w_idx in msg_idx:								
			word_counts[w_idx]+=1	

print "training files :", repr(train_files) 
print "usr2idx  files :",repr(usr2idx_files)
unigram_distribution = word_counts / word_counts.sum(0)
#pickle the word and user indices
print "pickling aux data..."
with open(aux_data,"wb") as fid:
	cPickle.dump([wrd2idx,unigram_distribution,E], fid, cPickle.HIGHEST_PROTOCOL)
