from pdb import set_trace

out = "DATA/embeddings/usr2vec_400_master_4.txt"
n_lines = 0
emb_size = 0
for i in xrange(1,8):
	with open("DATA/embeddings/usr2vec_400_%d.txt" % i, "r") as fid:
		_, emb_size = fid.readline().split()
		for line in fid:
			n_lines+=1

with open(out,"w") as fod:
	fod.write("%d %d\n" % (n_lines,int(emb_size)))

	for i in xrange(1,8):
		with open("DATA/embeddings/usr2vec_400_%d.txt" % i, "r") as fid:
			fid.readline()
			for line in fid:
				# print line
				fod.write(line)
