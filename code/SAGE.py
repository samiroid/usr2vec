import cPickle
import numpy as np
import theano
import theano.tensor as T

def init_W(size, rng):    
    W = np.asarray(rng.normal(0,0.01, size=size))
    return theano.shared(W.astype(theano.config.floatX), borrow=True)

class SAGE():

	def __init__(self, m_distribution, n_users, initial_etas=None, lrate=0.01):
		#initializations
		rng = np.random.RandomState(1234)        	
		if initial_etas is not None:
			self.user_etas = theano.shared(initial_etas.T.astype(theano.config.floatX))
		else:
			self.user_etas = init_W((len(m_distribution),n_users), rng)				
		m_distribution = theano.shared(m_distribution)		
		#model
		usr_idx = T.iscalar('usr')
		#document counts
		doc_counts = T.imatrix('doc_counts')		
		doc_sum = T.sum(doc_counts, 0)		
		#parameter vector
		eta = self.user_etas[:,usr_idx]				
		#objective
		dot_prod = T.dot(doc_counts.T,eta[:,None])		
		m_divergence = m_distribution + eta
		m_divergence_sum = T.sum(T.exp(m_divergence))				
		obj = dot_prod - doc_sum*T.log(m_divergence_sum)		
		obj = T.sum(obj)
		#gradient		
		grad = T.grad(obj, eta)
		#sparse gradient update
		grad_updt = T.set_subtensor(eta, eta+lrate*grad)
		updates = ((self.user_etas, grad_updt),)
		self.train = theano.function(inputs=[usr_idx,doc_counts],	
									 outputs=obj,
									 updates=updates
									 )
		#predict user given text
		user_params = self.user_etas + m_distribution[:,None]		
		pred_scores = T.dot(user_params.T,doc_counts)		
		y_hat = T.nnet.softmax(pred_scores.T).T[usr_idx].sum()		
		# y_hat = T.nnet.softmax(pred_scores.T).T[usr_idx].sum()		
		self.evaluate = theano.function(inputs=[usr_idx,doc_counts],
									   outputs=y_hat)		

	def save(self, path):
		with open(path,"wb") as fod:
			cPickle.dump(self.W.get_value(), fod, cPickle.HIGHEST_PROTOCOL)

class Sampler:
	def __init__(self, user_etas, m_distribution, usr2idx, wrd2idx):
		self.user_etas = user_etas
		self.m_distribution = m_distribution
		self.usr2idx = usr2idx
		self.wrd2idx = wrd2idx
		self.idx2wrd = {v:k for k,v in wrd2idx.items()}
		wrd_prbs = self.softmax(user_etas+m_distribution[:,None])
		self.new_wrd_prbs = self.softmax(m_distribution[:,None]-user_etas)
		# from pdb import set_trace; set_trace()
		#flip the columns to obtain the inverse probality 
		#(i.e., the word with the highest probability becomes the word with lowest probability)
		self.inv_wrd_prbs = np.flipud(wrd_prbs)

	def new_negative_samples(self, u_id, exclude=[], n_samples=1):
		samples = []		
		while len(samples) != n_samples:
			vals = np.random.multinomial(1, self.new_wrd_prbs[:,u_id])		
			wrd_idx = np.nonzero(vals)[0][0]
			if wrd_idx not in exclude: samples.append(wrd_idx)
		return samples

	def random_negative_samples(self, exclude=[], n_samples=1):		
		samples = []		
		while len(samples) != n_samples:			
			wrd_idx = np.random.randint(0,len(self.wrd2idx))
			if wrd_idx not in exclude: samples.append(wrd_idx)
		return samples

	def simple_negative_samples(self, exclude=[], n_samples=1):		
		samples = []		
		while len(samples) != n_samples:
			vals = np.random.multinomial(1, self.m_distribution)		
			wrd_idx = np.nonzero(vals)[0][0]
			if wrd_idx not in exclude: samples.append(wrd_idx)
		return samples

	def negative_samples(self, u_id, exclude=[], n_samples=1):
		samples = []		
		while len(samples) != n_samples:
			vals = np.random.multinomial(1, self.inv_wrd_prbs[:,u_id])		
			wrd_idx = np.nonzero(vals)[0][0]
			if wrd_idx not in exclude: samples.append(wrd_idx)
		return samples

	def samples(self, u_id, exclude=[], n_samples=1):
		samples = []		
		while len(samples) != n_samples:
			vals = np.random.multinomial(1, self.wrd_prbs[:,u_id])		
			wrd_idx = np.nonzero(vals)[0][0]
			if wrd_idx not in exclude: samples.append(wrd_idx)
		return samples

	def softmax(self, vec):
		return np.exp(vec)/np.sum(np.exp(vec),axis=0)
