import sys
from ipdb import set_trace
import numpy as np
import cPickle
from collections import Counter
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

clean_user_tweets, emb_path, stuff_pickle, training_data_path, sage_data_path = sys.argv[1:]

if "clean" not in clean_user_tweets:
	raise EnvironmentError("Requires \"clean\" file...")

MAX_WORDS = 20000
all_users =  {}
word_counter = Counter()
# max_msg_len = 0
n_docs=0
print "Building vocabulary..."
with open(clean_user_tweets,"r") as fid:	
	for line in fid:	
		usr = line.split("\t")[0] 		
		all_users[usr] = None
		message = line.split("\t")[1].decode("utf-8").split()
		word_counter.update(message)		
		#remember the max message length
		# if len(message) > max_msg_len: max_msg_len = len(message)
		n_docs+=1
		sys.stdout.write("\rdoc:%d" % n_docs)
		sys.stdout.flush()
		# all_messages.append(tokens)		
print ""
#build user index
usr2idx = {w:i for i,w in enumerate(all_users.keys())}
print "Found %d users in the corpus" % len(usr2idx)
#keep only the MAX_WORDS more frequent words
sw = sorted(word_counter.items(), key=lambda x:x[1],reverse=True)
top_words = {w[0]:None for w in sw[:MAX_WORDS]}
print "Extracting pre-trained word embeddings"
with open(emb_path) as fid:        
    # Get emb size
    _, emb_size = fid.readline().split()        
    for line in fid.readlines():                    
        items = line.split()
        wrd   = items[0]
        e = np.array(items[1:]).astype(float)            
        if wrd in top_words:
        	top_words[wrd] = e        
#I WANT TO KEEP ONLY THE WORDS THAT HAVE AN EMBEDDINGx
words_with_embedding = [i for i,j in top_words.items() if j is not None]
wrd2idx = {w:i for i,w in enumerate(words_with_embedding)}
#generate the embedding matrix
E = np.zeros((int(emb_size), len(wrd2idx)))   
for wrd,idx in wrd2idx.items(): E[:, idx] = top_words[wrd]
# wrd2idx = {w:i for i,w in enumerate(top_words.keys())}
# E = None
print "Building training data..."
word_counts = np.zeros(len(wrd2idx))
usr_wrd_counts = np.zeros((len(wrd2idx),len(usr2idx)))
prev_user = None
prev_user_data = []
#write train data
f_train = open(training_data_path,"wb") 
#write train data for SAGE (SAGE needs sparse matrices)
f_sage = open(sage_data_path,"wb") 
with open(clean_user_tweets,"r") as fid:		
	for j, line in enumerate(fid):	
		user = line.split("\t")[0] 		
		message = line.split("\t")[1].decode("utf-8").split()	
		u_idx = usr2idx[user] 
		if j==0: prev_user = u_idx #first user
		#convert to indices
		msg_idx = [wrd2idx[w] for w in message if w in wrd2idx]
		if len(msg_idx)==0: continue			
		#compute background and user word distributions (for SAGE)
		for w_idx in msg_idx:								
			word_counts[w_idx]+=1	
			usr_wrd_counts[w_idx,u_idx]+=1
		#after accumulating all samples for current user
		#shuffle and write them to disk
		if u_idx != prev_user:
			#shuffle the data					
			rng.shuffle(prev_user_data)
			split = int(len(prev_user_data)*.9)
			train = prev_user_data[:split]
			test  = prev_user_data[split:]				
			stPickle.s_dump_elt([prev_user, train, test, [],[]], f_train)
			#save data as sparse matrices for SAGE
			train_matrix = as_sparse_matrix(train, len(wrd2idx))	
			test_matrix  = as_sparse_matrix(test, len(wrd2idx))		
			stPickle.s_dump_elt([prev_user, train_matrix, test_matrix], f_sage)				
			prev_user_data = []
		elif j == n_docs-1:	
			#take into account the very last message	
			prev_user_data.append(msg_idx)
			#shuffle the data		
			rng.shuffle(prev_user_data)
			split = int(len(prev_user_data)*.9)				
			train = prev_user_data[:split]
			test  = prev_user_data[split:]
			stPickle.s_dump_elt([prev_user, train, test, [],[]], f_train)	
			#save data as sparse matrices for SAGE
			train_matrix = as_sparse_matrix(train, len(wrd2idx))	
			test_matrix  = as_sparse_matrix(test, len(wrd2idx))		
			stPickle.s_dump_elt([prev_user, train_matrix, test_matrix], f_sage)				
		prev_user = u_idx
		prev_user_data.append(msg_idx)
print "Computing background word distributions (for SAGE)"
word_probs = word_counts / word_counts.sum(0)
#divide along axis to get likelihoods
# set_trace()
usr_lang_model = usr_wrd_counts / usr_wrd_counts.sum(0)[np.newaxis:,]

#pickle the word and user indices
print "Pickling stuff..."
with open(stuff_pickle,"wb") as fid:
	cPickle.dump([wrd2idx,usr2idx,word_probs,usr_lang_model,E], fid, cPickle.HIGHEST_PROTOCOL)










