from ipdb import set_trace
import sys
import cPickle
from gensim.models import Word2Vec
aux_pkl, user_embs_pkl, user_embs_txt = sys.argv[1:]

with open(aux_pkl,"r") as fid:
	_,usr2idx,_,_,_ = cPickle.load(fid)
	# zz = cPickle.load(fid)

with open(user_embs_pkl,"r") as fid:
	U = cPickle.load(fid)[0]

with open(user_embs_txt,"w") as fod:
	fod.write("%d %d\n" % (U.shape[1],U.shape[0]))	
	for user, u_id in usr2idx.items():
		# set_trace()
		emb = U[:,u_id]
		fod.write("%s %s\n" % (user, " ".join(map(str, emb))))

