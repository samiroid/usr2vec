import argparse
from bst import BinarySearchTree
import math
import os
from pdb import set_trace
import streaming_pickle as stPickle

"""
Module to reorganize training data
1. Sorts users by number their number of documents
2. Splits users into a set of files
"""

def get_parser():
    parser = argparse.ArgumentParser(description="Sort and split User2Vec training data")
    parser.add_argument('-input', type=str, required=True, help='train file(s)')
    parser.add_argument('-n_splits', type=int, help='number of splits',default=2)
    return parser

if __name__ == "__main__":
	parser = get_parser()
	args = parser.parse_args()	
	bst = BinarySearchTree(sort_key=lambda x:x[1])
	assert args.n_splits >= 2
	print "sorting users by #tweets..."	
	tf = open(args.input,"r")			
	for x in stPickle.s_load(tf):
		user, train, _, _,_ = x		
		bst.insert((user,len(train)))
	sorted_values = list(bst.values(reverse=True))
	sorted_users  = [x[0] for x in sorted_values]	
	print "[spliting into #files: %d]" % args.n_splits
	out_files = []	
	out_path, ext = os.path.splitext(args.input) 	
	for i in xrange(args.n_splits):		
		fname = "%s%d%s" % (out_path,i+1,ext)		
		print "   > %s" % fname
		f = open(fname,"w")
		out_files.append(f)
	tf.seek(0)
	out_log =  [[]]*args.n_splits
	# set_trace()
	print "[processing users]"
	partition_size = math.floor(len(sorted_users)*1.0/args.n_splits)
	for x in stPickle.s_load(tf):
		user, train, _, _,_ = x		
		user_rank = sorted_users.index(user)		
		fnumber   = int(math.floor(user_rank*1.0/partition_size)) 		
		if fnumber < 0: fnumber = 0		
		if fnumber > args.n_splits-1: fnumber = args.n_splits-1	
		print "   > user: %s | #train: %d | rank: %d | fnum: %d" % (user, len(train), user_rank, fnumber)
		out_file = out_files[fnumber]
		stPickle.s_dump_elt(x, out_file)
		out_log[fnumber].append(len(train))
	print "[removing original training file: %s]" % args.input
	os.remove(args.input)