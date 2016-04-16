import sys
from my_utils import preprocess 
from ipdb import set_trace

user_tweets, clean_file, mode, n_users = sys.argv[1:]

mode = mode.upper()
assert mode in ["SMALL","ALL"]

###### HACKS TO DO PARTIAL PROCESSING
prev_user=None
message_count=0
MAX_PER_USER = 100
MAX_USERS = 10
all_users =  {}
###### ###### ###### 
n_docs=0
print "Reading and preprocessing %s data..." % mode

file_counter = 0
user_counter = {}
fod = open(clean_file.replace(".txt","_"+str(file_counter))+".txt","w")

with open(user_tweets,"r") as fid:		
	for line in fid:	
		usr = line.split("\t")[0] 	
		user_counter[usr]=None
		if len(user_counter) > int(n_users):
			# set_trace()
			user_counter = {}
			file_counter+=1
			fod.close()
			fod = open(clean_file.replace(".txt","_"+str(file_counter))+".txt","w")
		if mode == "SMALL":
			# ###### HACKS TO DO PARTIAL PROCESSING
			if prev_user == usr:
				message_count+=1
			else:						
				message_count = 0
			prev_user = usr
			if message_count>=MAX_PER_USER: continue
			all_users[usr] = None
			if len(all_users)>MAX_USERS: 
				del all_users[usr]
				break			
			###### ###### ###### 
		message = line.split("\t")[1]		
		message = preprocess(message.decode("utf-8"))		
		#discard isolated chars        
		tokens = [a for a in message.split() if len(a) > 1]
		# word_counts.update(tokens)
		fod.write("%s\t%s\n" % (usr, ' '.join(tokens).encode("utf-8")))
		n_docs+=1
		sys.stdout.write("\rdoc:%d" % n_docs)
		sys.stdout.flush()
			# all_messages.append(tokens)	
fod.close()
print "\nWrote preprocessed file: %s with %d docs" % (clean_file,n_docs)