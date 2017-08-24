from ipdb import set_trace
import sys
import cPickle

user_embs_pkl, usr2idx_path,  user_embs_txt = sys.argv[1:]

with open(usr2idx_path, "r") as fid:
	usr2idx = cPickle.load(fid)	
with open(user_embs_pkl,"r") as fid:
	U = cPickle.load(fid)[0]
with open(user_embs_txt,"w") as fod:
	fod.write("%d %d\n" % (U.shape[1],U.shape[0]))	
	for user, u_id in usr2idx.items():		
		emb = U[:,u_id]
		fod.write("%s %s\n" % (user, " ".join(map(str, emb))))

