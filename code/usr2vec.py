import cPickle  
# from ipdb import set_trace
import numpy as np
import theano
import theano.tensor as T

def init_weight(rng, size):        
    return np.asarray(rng.normal(0,0.01, size=size))
    

def init_w2v_gauss(rng, E, n_users):
    mu  = np.mean(E,axis=1)
    mu  = np.squeeze(np.asarray(mu))
    cov = np.cov(E,rowvar=1)
    return np.random.multivariate_normal(mu, cov,size=n_users).T

def init_w2v_mean(rng, E, n_users):
    mu  = np.mean(E,axis=1)    
    U   = np.asarray(rng.normal(0,0.01, size=(E.shape[0],n_users)))    
    return U + mu[:,None]
    

class Usr2Vec():
      
  def __init__(self, E, n_users, lrate=0.0001, margin_loss=1, rng=None, init_w2v=False):

    # Generate random seed if not provided
    if rng is None:
      rng=np.random.RandomState(1234)            
    #parameters
    if init_w2v == "gauss":        
        U = init_w2v_gauss(rng, n_users, E)    
    elif init_w2v == "mean":        
        U = init_w2v_mean(rng, E, n_users)
    else:
        U = init_weight(rng, (E.shape[0],n_users))            
    U = theano.shared(U.astype(theano.config.floatX), borrow=True)
    E = theano.shared(E.astype(theano.config.floatX), borrow=True)     
    
    self.params      = [U]         
    self.margin_loss = margin_loss
    self.lrate       = lrate
    #input
    usr_idx      = T.iscalar('usr')    
    sent_idx     = T.ivector('sent')    
    neg_samp_idx = T.imatrix('neg_sample')
    # word_probs   = T.fvector('word_probs')     
    word_probs   = T.fscalar('word_probs')     
    curr_lrate   = T.fscalar('lrate')
    #embedding lookup
    usr         = U[:, usr_idx]
    sent        = E[:, sent_idx] 
    neg_samples = E[:, neg_samp_idx]
    #loss
    # objectives, _ = theano.scan(fn=self.rank_loss,
    #                             outputs_info=None,
    #                             sequences=[sent_idx,neg_samp_idx],                                    
    #                             non_sequences=[usr,E,U])
    pos_score = T.dot(usr,sent)
    neg_score = T.tensordot(usr,neg_samples,axes=(0,0))
    loss      = T.maximum(0, self.margin_loss - pos_score[:,None] + neg_score)
    # final_loss = loss.sum(axis=None) + word_probs.sum()
    final_loss = loss.sum(axis=None) + word_probs
    #Gradient wrt to user embeddings
    usr_grad = T.grad(final_loss, usr)
    #Sparse update
    upd_usr = T.set_subtensor(usr, usr - curr_lrate*usr_grad)
    updates = ((U, upd_usr),)
    # self.dbg = theano.function(inputs=[usr_idx, sent_idx, neg_samp_idx],
    #                              outputs=[usr,sent,neg_samples],      
    #                              mode="FAST_COMPILE")
    self.dbg = theano.function(inputs=[usr_idx, sent_idx, neg_samp_idx],
                                 outputs=[usr,sent,neg_samples])
    
    self.train = theano.function(inputs=[usr_idx, sent_idx, neg_samp_idx, word_probs,curr_lrate],
                                 outputs=final_loss,
                                 updates=updates,
                                 mode="FAST_RUN")
    #\propto P(message|usr)    
    # scores_m = T.exp(T.dot(U.T,E[:,sent_idx]))    
    scores_m = T.dot(U.T,E[:,sent_idx])    
    prob = T.nnet.softmax(scores_m.T).T
    log_prob = T.log(prob).sum(axis=1)
    #sum the scores for all the words    
    # scores_m = scores_m.sum(axis=1)
    # user_score = scores_m[usr_idx]
    user_score = log_prob[usr_idx]
    self.predict = theano.function(inputs=[usr_idx,sent_idx],
                                   outputs=[user_score,prob])    

  def rank_loss(self, w_idx, negs_idx, usr, E, U):
    w_emb     = E[:, w_idx]
    w_neg_emb = E[:, negs_idx]
    pos_score = T.dot(usr,w_emb)
    neg_score = T.dot(usr,w_neg_emb)
    loss      = T.maximum(0, self.margin_loss - pos_score + neg_score) 
    return loss 

  def save_model(self, path):
    model = [self.params[i].get_value() for i in range(len(self.params))]
    with open(path,"wb") as fid:
      cPickle.dump(model,fid,cPickle.HIGHEST_PROTOCOL)