from collections import defaultdict
from ipdb import set_trace
import os
import sys

sys.path.append("code")
from sma_toolkit.preprocess import preprocess 

#command line arguments
path_in, path_out, mode = sys.argv[1:]
mode = mode.upper()
assert mode in ["SMALL","ALL"]

################## 
# settings for SMALL mode (i.e process only a subset of the data)
MAX_PER_USER = 10
MAX_USERS = 10
n_docs=0
#
################## 
tweets_by_user = defaultdict(list)
print "Reading and preprocessing %s data..." % mode
with open(path_in,"r") as fid:		
	for line in fid:	
		user = line.split("\t")[0] 			
		message = line.split("\t")[1]		
		message = preprocess(message.decode("utf-8"))				
		#partial processing 
		if mode == "SMALL":
			#keep only MAX_PER_USER messages per user
			if len(tweets_by_user[user]) > MAX_PER_USER:
				continue
			#keep only MAX_USERS users
			if len(tweets_by_user) > MAX_USERS:
				#remove the last user because it only has one message	
				del tweets_by_user[user]
				print "\n[max users: %d]" % len(tweets_by_user)
				break
		tweets_by_user[user].append(message)
		n_docs+=1
		sys.stdout.write("\rdoc #%d" % n_docs)
		sys.stdout.flush()		

if not os.path.exists(os.path.dirname(path_out)):
	os.makedirs(os.path.dirname(path_out))

with open(path_out,"w") as fod:
	for user, messages in tweets_by_user.items():
		for m in messages:
			fod.write("%s\t%s\n" % (user, m.encode("utf-8")))
