'''
Deep actor-critic network, 
From "Continuous control with deep reinforcement learning", by Lillicrap et al, arXiv:1509.02971
'''

from pend import Pendulum
import tensorflow as tf
import numpy as np
import tflearn
import random
from collections import deque
import time
import signal
import matplotlib.pyplot as plt
from pinocchio.utils import *
from numpy.linalg import norm

### --- Random seed
RANDOM_SEED = int((time.time()%10)*1000)
print "Seed = %d" %  RANDOM_SEED
np .random.seed     (RANDOM_SEED)
tf .set_random_seed (RANDOM_SEED)
random.seed         (RANDOM_SEED)
n_init              = tflearn.initializations.truncated_normal(seed=RANDOM_SEED)
u_init              = tflearn.initializations.uniform(minval=-0.003, maxval=0.003,\
                                                      seed=RANDOM_SEED)

BATCH_SIZE          = 64            # Number of points to be fed in stochastic gradient
NH1 = NH2           = 250           # Hidden layer size
LEARNING_RATE       = 0.001

NX                  = 2             # ... training converges with q,qdot with 2x more neurones.
NU                  = 1             # Control is dim-1: joint torque
UMAX                = 2

RESTORE             = 'netvalues/fromacado'

### --- Q-value and policy networks
class PolicyNetwork:
    def __init__(self):
        nvars           = len(tf.trainable_variables())

        x               = tflearn.input_data(shape=[None, NX])
        net             = tflearn.fully_connected(x,   NH1, activation='relu', weights_init=n_init)
        net             = tflearn.fully_connected(net, NH2, activation='relu', weights_init=n_init)
        policy          = tflearn.fully_connected(net, NU,  activation='tanh', weights_init=u_init)*UMAX

        self.x          = x                                     # Network input <x> in Pi(x)
        self.policy     = policy                                # Network output <Pi>
        self.variables = tf.trainable_variables()[nvars:]       # Variables to be trained

    def setupOptim(self):

        uref            = tf.placeholder(tf.float32, [None, NU])
        loss            = tflearn.mean_square(uref, self.policy)
        optim           = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

        self.optim      = optim         # Optimizer
        self.uref       = uref
        self.loss       = loss
        return self

class QValueNetwork:
    def __init__(self):
        nvars           = len(tf.trainable_variables())

        x               = tflearn.input_data(shape=[None, NX])
        net             = tflearn.fully_connected(x,   NH1, activation='relu', weights_init=n_init)
        net             = tflearn.fully_connected(net, NH2, activation='relu', weights_init=n_init)
        qvalue          = tflearn.fully_connected(net, 1,   activation='relu', weights_init=n_init)

        self.x          = x                                     # Network input <x> in Pi(x)
        self.qvalue     = qvalue                                # Network output <Pi>
        self.variables = tf.trainable_variables()[nvars:]       # Variables to be trained

    def setupOptim(self):

        qref            = tf.placeholder(tf.float32, [None, 1 ])
        loss            = tflearn.mean_square(qref, self.qvalue)
        optim           = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

        self.optim      = optim         # Optimizer
        self.qref       = qref
        self.loss       = loss
        return self


### --- Tensor flow initialization

policy          = PolicyNetwork(). setupOptim()
qvalue          = QValueNetwork(). setupOptim()
sess            = tf.InteractiveSession()
tf.global_variables_initializer().run()

if len(RESTORE)>0:
    print "*** Restore net weights from ",RESTORE
    tf.train.Saver().restore(sess, RESTORE)

filename = 'databasexx.np' #'database.np'
D = np.load(open(filename))
#D=D[::5,:]
idx=np.nonzero(D[:,3]<100)[0]
D=D[idx,:]

# Learn policy
for i in range(1000000):
    batch = np.array(random.sample(D,BATCH_SIZE))
    x_batch = batch[:,:2]
    u_batch = batch[:,2:3]
    q_batch = batch[:,3:]
    sess.run(policy.optim, feed_dict={ policy.x    : x_batch,
                                       policy.uref : u_batch })
    sess.run(qvalue.optim, feed_dict={ qvalue.x    : x_batch,
                                       qvalue.qref : q_batch })
    if not i%100: print "Learn session <%d>: pi_loss=%.3f q_loss=%.3f" \
            %(i,
              sess.run(policy.loss, feed_dict={ policy.x    : x_batch,
                                                policy.uref : u_batch }),
              sess.run(qvalue.loss, feed_dict={ qvalue.x    : x_batch,
                                                qvalue.qref : q_batch }))

print "Sample points from nets"

U,Q = sess.run( [policy.policy,qvalue.qvalue], 
                feed_dict={ policy.x    : D[idx,:2],
                            qvalue.x    : D[idx,:2] })

import matplotlib.pyplot as plt
import pylab
from mpl_toolkits.mplot3d import Axes3D
plt.ion()

idx=random.sample(range(len(D)),1000)

fig = pylab.figure()
ax = Axes3D(fig)
ax.scatter(D[idx,0],D[idx,1],Q[idx],c='r')
ax.scatter(D[idx,0],D[idx,1],D[idx,3],c='b')

env = Pendulum(1)
env.DT              = .15
env.NDT             = 2
NSTEPS              = 30
env.Kf = 0.2
env.vmax = 100

def closest_u(x):
    i=np.argmin([ norm(d-x.T) for d in D[:,:2] ])
    return D[i,:]

def rollout(x0=None,NSTEPS=30):
    env.reset(x0)
    hx = []; hu = []
    for i in range(NSTEPS):
        hx.append( env.x.copy().T )
        u = sess.run(policy.policy, feed_dict={ policy.x    : env.x.T })
        #u = closest_u(env.x)[2:3]
        #print env.x.T,u
        hu.append( u.copy() )
        env.step(u)
        env.render()


