'''
Deep actor-critic network, classes defining the nets.
From "Continuous control with deep reinforcement learning", by Lillicrap et al, arXiv:1509.02971
'''

import tensorflow as tf
import numpy as np
import tflearn
import random


class QValueNetwork:
    def __init__(self):
        nvars           = len(tf.trainable_variables())

        x       = tflearn.input_data(shape=[None, NX])
        u       = tflearn.input_data(shape=[None, NU])

        netx1   = tflearn.fully_connected(x,     NH1, weights_init=n_init, activation='relu')
        netx2   = tflearn.fully_connected(netx1, NH2, weights_init=n_init)
        netu1   = tflearn.fully_connected(u,     NH1, weights_init=n_init, activation='linear')
        netu2   = tflearn.fully_connected(netu1, NH2, weights_init=n_init)
        net     = tflearn.activation     (netx2+netu2,activation='relu')
        qvalue  = tflearn.fully_connected(net,   1,   weights_init=u_init)

        self.x          = x                                # Network state   <x> input in Q(x,u)
        self.u          = u                                # Network control <u> input in Q(x,u)
        self.qvalue     = qvalue                           # Network output  <Q>
        self.variables  = tf.trainable_variables()[nvars:] # Variables to be trained
        self.hidens = [ netx1, netx2, netu1, netu2 ]       # Hidden layers for debug

    def setupOptim(self):
        qref            = tf.placeholder(tf.float32, [None, 1])
        loss            = tflearn.mean_square(qref, self.qvalue)
        optim           = tf.train.AdamOptimizer(QVALUE_LEARNING_RATE).minimize(loss)
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

class PolicyNetwork:
    def __init__(self):
        nvars           = len(tf.trainable_variables())

        x               = tflearn.input_data(shape=[None, NX])
        net             = tflearn.fully_connected(x,   NH1, activation='relu', weights_init=n_init)
        net             = tflearn.fully_connected(net, NH2, activation='relu', weights_init=n_init)
        policy          = tflearn.fully_connected(net, NU,  activation='tanh', weights_init=u_init)*env.umax

        self.x          = x                                     # Network input <x> in Pi(x)
        self.policy     = policy                                # Network output <Pi>
        self.variables = tf.trainable_variables()[nvars:]       # Variables to be trained

    def setupOptim(self):

        qgradient       = tf.placeholder(tf.float32, [None, NU])  
        grad            = tf.gradients(self.policy, self.variables, -qgradient)
        optim           = tf.train.AdamOptimizer(POLICY_LEARNING_RATE).\
            apply_gradients(zip(grad,self.variables))

        self.qgradient  = qgradient     # Q-value gradient wrt control (input value) 
        self.optim      = optim         # Optimizer
        return self

    def setupTargetAssign(self,nominalNet,tau=UPDATE_RATE):
        self.update_variables = \
            [ target.assign( tau*ref + (1-tau)*target )  \
                  for target,ref in zip(self.variables,nominalNet.variables) ]
        return self


from collections import namedtuple
FileData = namedtuple('FileData', 'date robottype Kf nsteps nepisodes dt ndt vmax umax')
def fileDateFromEnv(env,robotname, NSTEPS,NEPISODES):
    return FileData(time.ctime(),
                    robotname,env.Kf, NSTEPS,NEPISODES, env.DT, env.NDT, env.vmax, env.umax)
# fileDateFromEnv(env,"simple pendulum",NSTEPS,NEPISODES)
                    
weightFiles = {}

weightFiles["netvalues/actorcritic"] \
    = FileData(date='Wed Jul  5 09:32:56 2017', robottype='simple pendulum', 
               Kf=0.2, nsteps=30, nepisodes=5000, dt=0.15, ndt=2, vmax=100, umax=2.0)

                        
