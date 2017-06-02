'''
Deep actor-critic network, 
From "Continuous control with deep reinforcement learning", by Lillicrap et al, arXiv:1509.02971
'''

from pendulum import Pendulum
import tensorflow as tf
import numpy as np
import tflearn
import random
from collections import deque
import time
import signal
import matplotlib.pyplot as plt

### --- Random seed
RANDOM_SEED = int((time.time()%10)*1000)
print "Seed = %d" %  RANDOM_SEED
np .random.seed     (RANDOM_SEED)
tf .set_random_seed (RANDOM_SEED)
random.seed         (RANDOM_SEED)
n_init              = tflearn.initializations.truncated_normal(seed=RANDOM_SEED)
u_init              = tflearn.initializations.uniform(minval=-0.003, maxval=0.003,\
                                                      seed=RANDOM_SEED)

### --- Hyper paramaters
NEPISODES               = 5000           # Max training steps
NSTEPS                  = 100           # Max episode length
QVALUE_LEARNING_RATE    = 0.001         # Base learning rate for the Q-value Network
POLICY_LEARNING_RATE    = 0.0001        # Base learning rate for the policy network
DECAY_RATE              = 0.99          # Discount factor 
UPDATE_RATE             = 0.01          # Homotopy rate to update the networks
REPLAY_SIZE             = 10000         # Size of replay buffer
BATCH_SIZE              = 64            # Number of points to be fed in stochastic gradient
NH1 = NH2               = 250           # Hidden layer size
RESTORE                 = ""#"netvalues/actorcritic.25.ckpt" # Previously optimize net weight 
                                        # (set empty string if no)
RENDERRATE              = 100           # Render rate (rollout and plot) during training (0 = no)
REGULAR                 = True          # Render on a regular grid vs random grid


### --- Environment
env                 = Pendulum(1)       # Continuous pendulum
env.withSinCos      = True              # State is dim-3: (cosq,sinq,qdot) ...
NX                  = env.nobs          # ... training converges with q,qdot with 2x more neurones.
NU                  = env.nu            # Control is dim-1: joint torque

env.DT              = .15
env.NDT             = 2
env.Kf              = 0.1
NSTEPS              = 30
#DECAY_RATE          = 1

env.Kf = 0.2
env.vmax = 100
#RESTORE                 = "netvalues/actorcritic.15.kf2" # Previously optimize net weight 
RESTORE                 = "netvalues/actorcritic.15.kf2.25000" # Previously optimize net weight 


### --- Q-value and policy networks

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

### --- Replay memory
class ReplayItem:
    def __init__(self,x,u,r,d,x2):
        self.x          = x
        self.u          = u
        self.reward     = r
        self.done       = d
        self.x2         = x2

replayDeque = deque()

### --- Tensor flow initialization

policy          = PolicyNetwork(). setupOptim()
policyTarget    = PolicyNetwork(). setupTargetAssign(policy)

qvalue          = QValueNetwork(). setupOptim()
qvalueTarget    = QValueNetwork(). setupTargetAssign(qvalue)

sess            = tf.InteractiveSession()
tf.global_variables_initializer().run()

if len(RESTORE)>0:
    print "*** Restore net weights from ",RESTORE
    tf.train.Saver().restore(sess, RESTORE)

def rendertrial(maxiter=NSTEPS,verbose=True):
    x = env.reset()
    rsum = 0.
    for i in range(maxiter):
        u = sess.run(policy.policy, feed_dict={ policy.x: x.T })
        x, reward = env.step(u)
        env.render()
        time.sleep(1e-2)
        rsum += reward
    if verbose: print 'Lasted ',i,' timestep -- total reward:',rsum
signal.signal(signal.SIGTSTP, lambda x,y:rendertrial()) # Roll-out when CTRL-Z is pressed

### History of search
h_rwd = []
h_qva = []
h_ste = []    

# Mesh grid for rendering the policy and value functions.
if REGULAR:  # Regular sampling
    X = np.vstack([ x.ravel() for x in np.meshgrid(np.arange(env.qlow,env.qup,.3),
                                                   np.arange(env.vlow,env.vup,.3)) ]).T
else: # Random sampling
    X=np.vstack([ np.diag([2*np.pi,16])*rand(2)+np.matrix([-np.pi,-8]) for i in range(1000) ])

### --- Training
for episode in range(1,NEPISODES):
    x    = env.reset().T
    rsum = 0.0

    for step in range(NSTEPS):
        u       = sess.run(policy.policy, feed_dict={ policy.x: x }) # Greedy policy ...
        u      += 1. / (1. + episode + step)                         # ... with noise
        x2,r    = env.step(u)
        x2      = x2.T
        done    = False                                              # pendulum scenario is endless.

        replayDeque.append(ReplayItem(x,u,r,done,x2))                # Feed replay memory ...
        if len(replayDeque)>REPLAY_SIZE: replayDeque.popleft()       # ... with FIFO forgetting.

        rsum   += r
        if done or np.linalg.norm(x-x2)<1e-3: break                  # Break when pendulum is still.
        x       = x2

        # Start optimizing networks when memory size > batch size.
        if len(replayDeque) > BATCH_SIZE:     
            batch = random.sample(replayDeque,BATCH_SIZE)            # Random batch from replay memory.
            x_batch    = np.vstack([ b.x      for b in batch ])
            u_batch    = np.vstack([ b.u      for b in batch ])
            r_batch    = np.vstack([ b.reward for b in batch ])
            d_batch    = np.vstack([ b.done   for b in batch ])
            x2_batch   = np.vstack([ b.x2     for b in batch ])

            # Compute Q(x,u) from target network
            u2_batch   = sess.run(policyTarget.policy, feed_dict={ policyTarget .x : x2_batch})
            q2_batch   = sess.run(qvalueTarget.qvalue, feed_dict={ qvalueTarget.x : x2_batch,
                                                                   qvalueTarget.u : u2_batch })
            qref_batch = r_batch + (d_batch==False)*(DECAY_RATE*q2_batch)

            # Update qvalue to solve HJB constraint: q = r + q'
            sess.run(qvalue.optim, feed_dict={ qvalue.x    : x_batch,
                                               qvalue.u    : u_batch,
                                               qvalue.qref : qref_batch })

            # Compute approximate policy gradient ...
            u_targ  = sess.run(policy.policy,   feed_dict={ policy.x        : x_batch} )
            qgrad   = sess.run(qvalue.gradient, feed_dict={ qvalue.x        : x_batch,
                                                            qvalue.u        : u_targ })
            # ... and take an optimization step along this gradient.
            sess.run(policy.optim,feed_dict= { policy.x         : x_batch,
                                               policy.qgradient : qgrad })

            # Update target networks by homotopy.
            sess.run(policyTarget. update_variables)
            sess.run(qvalueTarget.update_variables)

    # \\\END_FOR step in range(NSTEPS)

    # Display and logging (not mandatory).
    maxq = np.max( sess.run(qvalue.qvalue,feed_dict={ qvalue.x : x_batch,
                                                      qvalue.u : u_batch }) ) \
                                                      if 'x_batch' in locals() else 0
    print 'Ep#{:3d}: lasted {:d} steps, reward={:3.0f}, max qvalue={:2.3f}' \
        .format(episode, step,rsum, maxq)
    h_rwd.append(rsum)
    h_qva.append(maxq)
    h_ste.append(step)
    if not (episode+1) % renderrate:     
        # Rollout and render in Gepetto
        rendertrial(100)
        # Generate sampling of the policy/value functions
        U = sess.run(policy.policy, feed_dict={ policy.x: env.obs(X.T).T })
        Q = sess.run(qvalue.qvalue, feed_dict={ qvalue.x: env.obs(X.T).T,
                                                qvalue.u: U })
        # Scatter plot of policy/value funciton sampling (in file)
        plt.clf()
        plt.subplot(1,2,1)
        plt.scatter(X[:,0],X[:,1],c=U[:],s=50,linewidths=0,alpha=.8,vmin=-2,vmax=2)
        plt.colorbar()
        plt.subplot(1,2,2)
        plt.scatter(X[:,0],X[:,1],c=Q[:],s=50,linewidths=0,alpha=.8)
        plt.colorbar()
        plt.savefig('figs/actorcritic_%d.png' % episode)

# \\\END_FOR episode in range(NEPISODES)

if NEPISODES>0:
    print "Average reward during trials: %.3f" % (sum(h_rwd)/(NEPISODES+1))
    plt.plot( np.cumsum(h_rwd)/range(1,NEPISODES) )
    plt.show()

#rendertrial()

