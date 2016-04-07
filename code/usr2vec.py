import cPickle  
# from ipdb import set_trace
import numpy as np
import theano
import theano.tensor as T

def init_weight(rng, size):
        
    W = np.asarray(rng.normal(0,0.01, size=size))
    return theano.shared(W.astype(theano.config.floatX), borrow=True)

class Usr2Vec():
      
  def __init__(self, E, n_users, lrate=0.0001, margin_loss=1, rng=None):

    # Generate random seed if not provided
    if rng is None:
      rng=np.random.RandomState(1234)            
    #parameters
    U = init_weight(rng, (E.shape[0],n_users))    
    E = theano.shared(value=E.astype(theano.config.floatX), 
                            borrow=True)     
    
    self.params      = [U]         
    self.margin_loss = margin_loss
    #input
    usr_idx      = T.iscalar('usr')    
    sent_idx     = T.ivector('sent')    
    neg_samp_idx = T.imatrix('neg_sample')
    word_probs   = T.fvector('word_probs')     
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
    final_loss = loss.sum(axis=None) + word_probs.sum()
    #Gradient wrt to user embeddings
    usr_grad = T.grad(final_loss, usr)
    #Sparse update
    upd_usr = T.set_subtensor(usr, usr - lrate*usr_grad)
    updates = ((U, upd_usr),)
    self.train = theano.function(inputs=[usr_idx, sent_idx, neg_samp_idx, word_probs],
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