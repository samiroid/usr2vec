# -*- coding: utf-8 -*-
# Read historical tweets
import csv	
from ipdb import set_trace
import sys
import ast

csv.field_size_limit(sys.maxsize)

with open("/Users/samir/Dev/projects/tweet_sarcasm/tweet_sarcasm/DATA/txt/bamman.csv") as fid:	
	usr_dict = {line.split()[0].replace('"',""):line.split()[2] for line in fid} 
	

with open('DATA/txt/All_historical_tweets.csv', 'rb') as f:  
	with open("DATA/txt/historical_tweets.txt","w") as o:
		fieldnames = ['tweet_id', 'historical_tweets']
		reader = csv.DictReader(f)
		fieldnames = reader.fieldnames
		print fieldnames
		for row in reader:
			# set_trace()
			tweet_id = row[fieldnames[0]]
			historical_tweets = row[fieldnames[1]].decode("utf-8")

			ht = ast.literal_eval(historical_tweets)
			for m in ht:
				clean_m = m.replace(u"“","").replace(u"”","").replace("'","").replace("\"","").replace("\n","")
				o.write("%s\t%s\n" % (usr_dict[tweet_id],clean_m.encode("utf-8")))
			

			# print historical_tweets
			# break

