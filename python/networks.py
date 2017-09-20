import tensorflow as tf
import numpy as np
import tflearn
import random

class WeightInit:
    def __init__(self,seed):
        self.reset(seed)
    def reset(self,seed):
        self.seed = seed
        self.n  = tflearn.initializations.truncated_normal(seed=seed)
        self.u  = tflearn.initializations.uniform(minval=-0.003, maxval=0.003,\
                                                      seed=seed)
    
winit           = WeightInit(1)
UPDATE_RATE     = 5e-3


class PolicyNetwork:
    '''
    Simple network from X state to U state, where X and U may or may not have the same dimension.
    '''
    def __init__(self,NX,NU,NH1=250,NH2=None,umax=None):
        if NH2 is None: NH2 = NH1
        self.NU = NU
        self.NX = NX
        
        nvars          = len(tf.trainable_variables())
        olay           = 'relu' if umax is None else 'tanh'

        x              = tflearn.input_data(shape=[None, NX])
        net            = tflearn.fully_connected(x,   NH1, activation='relu', weights_init=winit.n)
        net            = tflearn.fully_connected(net, NH2, activation='relu', weights_init=winit.n)
        policy         = tflearn.fully_connected(net, NU,  activation=olay  , weights_init=winit.u)

        self.x         = x                                      # Network input <x> in Pi(x)
        self.policy    = policy                                 # Network output <Pi>
        self.variables = tf.trainable_variables()[nvars:]       # Variables to be trained

        self.withUmax(umax)

    def withUmax(self,umax):
        if umax is None: 
            print 'umax none'
            return
        umax = np.matrix(umax)
        self.head = self.policy
        if   umax.shape == (1,1):  
            print 'umax float'
            self.policy = self.policy*umax[0,0]
        elif umax.shape == (1,2): 
            print 'umax pair'
            self.policy = (self.policy*(umax[0,1]-umax[0,0]) + umax[0,0] + umax[0,1])/2
        elif umax.shape == (2,self.NU): 
            print 'umax list'
            l,u = umax
            self.policy = (tf.multiply(self.policy,u-l) + l+u)/2

        # if isinstance(umax,float): self.policy = self.policy*umax
        # try: 
        #     if len(umax)==2: 
        #         self.pi     = self.policy
        #         if isinstance(umax[0],float):
        #             self.policy = (self.policy*(umax[1]-umax[0]) + umax[0] + umax[1]) / 2
        #         else:
        #             ul,uu = umax
        #             self.policy = (tf.multiply(self.pi,[ u-l for l,u in zip(ul,uu) ]) + ul+uu)/2
        # except: 
        #     pass

    def setupOptim(self,otype = 'actorcritic'):
        if otype == 'actorcritic': return self.setupActorCriticOptim()
        else:                      return self.setupDirectOptim()

    def setupActorCriticOptim(self,learningRate = UPDATE_RATE):
        qgradient       = tf.placeholder(tf.float32, [None, self.NU])  
        grad            = tf.gradients(self.policy, self.variables, -qgradient)
        optim           = tf.train.AdamOptimizer(learningRate).\
            apply_gradients(zip(grad,self.variables))

        self.qgradient  = qgradient     # Q-value gradient wrt control (input value) 
        self.optim      = optim         # Optimizer
        return self

    def setupDirectOptim(self,learningRate = UPDATE_RATE):
        uref            = tf.placeholder(tf.float32, [None, self.NU])
        loss            = tflearn.mean_square(uref, self.policy)
        optim           = tf.train.AdamOptimizer(learningRate).minimize(loss)

        self.optim      = optim         # Optimizer
        self.uref       = uref
        self.loss       = loss
        return self

    def setupTargetAssign(self,nominalNet,tau=UPDATE_RATE):
        self.update_variables = \
            [ target.assign( tau*ref + (1-tau)*target )  \
                  for target,ref in zip(self.variables,nominalNet.variables) ]
        return self


### --- Q-value and policy networks
class QValueNetwork:
    def __init__(self,NX,NU,NH1=250,NH2=None):
        if NH2 is None: NH2 = NH1

        nvars   = len(tf.trainable_variables())

        x       = tflearn.input_data(shape=[None, NX])
        u       = tflearn.input_data(shape=[None, NU])

        netx1   = tflearn.fully_connected(x,     NH1, weights_init=winit.n, activation='relu')
        netx2   = tflearn.fully_connected(netx1, NH2, weights_init=winit.n)
        netu1   = tflearn.fully_connected(u,     NH1, weights_init=winit.n, activation='linear')
        netu2   = tflearn.fully_connected(netu1, NH2, weights_init=winit.n)
        net     = tflearn.activation     (netx2+netu2,activation='relu')
        qvalue  = tflearn.fully_connected(net,   1,   weights_init=winit.u)

        self.x          = x                                # Network state   <x> input in Q(x,u)
        self.u          = u                                # Network control <u> input in Q(x,u)
        self.qvalue     = qvalue                           # Network output  <Q>
        self.variables  = tf.trainable_variables()[nvars:] # Variables to be trained
        self.hidens = [ netx1, netx2, netu1, netu2 ]       # Hidden layers for debug

    def setupOptim(self,learningRate = UPDATE_RATE):
        qref            = tf.placeholder(tf.float32, [None, 1])
        loss            = tflearn.mean_square(qref, self.qvalue)
        optim           = tf.train.AdamOptimizer(learningRate).minimize(loss)
        gradient        = tf.gradients(self.qvalue, self.u)[0] / float(BATCH_SIZE)

        self.qref       = qref          # Reference Q-values
        self.optim      = optim         # Optimizer
        self.gradient   = gradient      # Gradient of Q wrt the control  dQ/du (for policy training)
        return self

    def setupTargetAssign(self,nominalNet,tau=UPDATE_RATE):
        self.update_variables = \
            [ target.assign( tau*ref + (1-tau)*target )  \
                  for target,ref in zip(self.variables,nominalNet.variables) ]
        return self

