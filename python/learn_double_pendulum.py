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
NEPISODES               = 50000          # Max training steps
NSTEPS                  = 30            # Max episode length
QVALUE_LEARNING_RATE    = 0.001         # Base learning rate for the Q-value Network
POLICY_LEARNING_RATE    = 0.0001        # Base learning rate for the policy network
DECAY_RATE              = 0.99          # Discount factor 
UPDATE_RATE             = 0.01          # Homotopy rate to update the networks
REPLAY_SIZE             = 10000         # Size of replay buffer
BATCH_SIZE              = 128           # Number of points to be fed in stochastic gradient
NH1 = NH2               = 250           # Hidden layer size
RESTORE                 = ""#"netvalues/actorcritic.15.kf2" # Previously optimize net weight 
                                        # (set empty string if no)
RENDERRATE              = 20           # Render rate (rollout and plot) during training (0 = no)
#RENDERACTION            = [ 'saveweights',  'draw', 'rollout' ]
REGULAR                 = True          # Render on a regular grid vs random grid


from collections import namedtuple
Data = namedtuple('Data', [ 'x0', 'X', 'cost', 'U', 'T' ])

dataflat = np.load('data/planner/double/5_15/grid_refine.npy')
data=[]
for i,d in enumerate(dataflat): data.append(Data(*d))


### --- Environment

env = Pendulum(2,length=.5,mass=3.0,armature=10.)
env.withSinCos      = True              # State is dim-3: (cosq,sinq,qdot) ...
NX                  = env.nobs          # ... training converges with q,qdot with 2x more neurones.
NU                  = env.nu            # Control is dim-1: joint torque

env.DT              = 0.15
env.NDT             = 15
env.Kf              = 2.0 # 1.0
env.vmax            = 100
env.umax            = 15.
env.modulo          = True
NSTEPS              = 50
BATCH_SIZE          = 128
RESTORE             = ''   #'netvalues/double/actorcritic_double.59999'
RENDERACTION        = [ 'rollout', ] # 'draw', 'rollout'
RENDERRATE          = 2000         # Render rate (rollout and plot) during training (0 = no)


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

    def setupOptim(self,otype = 'actorcritic'):
        if otype == 'actorcritic': return self.setupActorCriticOptim()
        else:                      return self.setupDirectOptim()

    def setupActorCriticOptim(self):
        qgradient       = tf.placeholder(tf.float32, [None, NU])  
        grad            = tf.gradients(self.policy, self.variables, -qgradient)
        optim           = tf.train.AdamOptimizer(POLICY_LEARNING_RATE).\
            apply_gradients(zip(grad,self.variables))

        self.qgradient  = qgradient     # Q-value gradient wrt control (input value) 
        self.optim      = optim         # Optimizer
        return self

    def setupDirectOptim(self):
        uref            = tf.placeholder(tf.float32, [None, NU])
        loss            = tflearn.mean_square(uref, self.policy)
        optim           = tf.train.AdamOptimizer(POLICY_LEARNING_RATE).minimize(loss)

        self.optim      = optim         # Optimizer
        self.uref       = uref
        self.loss       = loss
        return self

    def setupTargetAssign(self,nominalNet,tau=UPDATE_RATE):
        self.update_variables = \
            [ target.assign( tau*ref + (1-tau)*target )  \
                  for target,ref in zip(self.variables,nominalNet.variables) ]
        return self

class ValueNetwork:
    def __init__(self):
        nvars           = len(tf.trainable_variables())

        x       = tflearn.input_data(shape=[None, NX])

        net     = tflearn.fully_connected(x,     NH1, weights_init=n_init, activation='relu')
        net     = tflearn.fully_connected(net,   NH2, weights_init=n_init, activation='relu')
        value   = tflearn.fully_connected(net,   1,   weights_init=u_init)

        self.x          = x                                # Network state   <x> input in Q(x,u)
        self.value      = value                            # Network output  <V>
        self.variables  = tf.trainable_variables()[nvars:] # Variables to be trained

    def setupOptim(self):
        vref            = tf.placeholder(tf.float32, [None, 1])
        loss            = tflearn.mean_square(vref, self.value)
        optim           = tf.train.AdamOptimizer(QVALUE_LEARNING_RATE).minimize(loss)

        self.vref       = vref          # Reference Q-values
        self.optim      = optim         # Optimizer
        return self

class StateNetwork:
    def __init__(self):
        nvars           = len(tf.trainable_variables())

        x       = tflearn.input_data(shape=[None, NX])

        net     = tflearn.fully_connected(x,     NH1, weights_init=n_init, activation='relu')
        net     = tflearn.fully_connected(net,   NH2, weights_init=n_init, activation='relu')
        x2      = tflearn.fully_connected(net,   NX,  weights_init=u_init)

        self.x          = x                                # Network state   <x> input in Q(x,u)
        self.x2         = x2                               # Network output  <x2>
        self.variables  = tf.trainable_variables()[nvars:] # Variables to be trained

    def setupOptim(self):
        x2ref           = tf.placeholder(tf.float32, [None, NX])
        loss            = tflearn.mean_square(x2ref, self.x2)
        optim           = tf.train.AdamOptimizer(POLICY_LEARNING_RATE).minimize(loss)

        self.x2ref      = x2ref         # Reference next state
        self.optim      = optim         # Optimizer
        return self

### --- Tensor flow initialization

policy          = PolicyNetwork(). setupOptim('direct')
value           = ValueNetwork() . setupOptim()
state           = StateNetwork() . setupOptim()

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
        x, reward = env.step(u.T)
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

    #grid = (np.arange(env.qlow,env.qup,.3),)*env.nq + \
    #    (np.arange(env.vlow,env.vup,.3),)*env.nv
    #X = np.vstack([ x.ravel() for x in np.meshgrid(*grid) ]).T
    
    X = np.vstack([ d.x0.T for d in data])
else: # Random sampling
    from pinocchio.utils import rand
    X=rand([10000,2])*np.diag([4*np.pi,16]) + np.matrix([-2*np.pi,-8]) 
#np.vstack([ np.diag([2*np.pi,16])*rand(2)+np.matrix([-np.pi,-8]) for i in range(1000) ])


### --- Replay memory
# class ReplayItem:
#     def __init__(self,x,u,r,d,x2):
#         self.x          = x
#         self.u          = u
#         self.reward     = r
#         self.done       = d
#         self.x2         = x2

ReplayItem = namedtuple('ReplayItem', 'x u reward done x2 value' )
ReplayItem.__new__.__defaults__ = ( None, )

replayDeque = deque()

### Data
for d in data:
    T = d.cost
    for x,u,t in zip(d.X,d.U,d.T):
        x2 = np.asarray(env.dynamics(np.matrix(x).T,np.matrix(u).T)[0].flat)
        o  = env.obs(np.matrix(x ).T).flat
        o2 = env.obs(np.matrix(x2).T).flat
        replayDeque.append( ReplayItem( x=o, u=u.copy(), reward=env.DT, done=False, 
                                        x2=o2, value = T-t ) )
        #if t>T*.9: break  # avoid trajectory ends
    #if len(replayDeque)>BATCH_SIZE: break
    replayDeque[-1] = replayDeque[-1]._replace(done=True)
    
print 'Done loading the motion lib'


from pinocchio.utils import zero
def o2x(o):
    x = zero(4)
    x[0] = np.arctan2(o[1],o[0])
    x[1] = np.arctan2(o[3],o[2])
    x[2:] = o[4:]
    return x

### --- Training
for episode in range(NEPISODES):
    print 'Episode #',episode
    batch = random.sample(replayDeque,BATCH_SIZE)            # Random batch from replay memory.
    x_batch    = np.vstack([ b.x      for b in batch ])
    u_batch    = np.vstack([ b.u      for b in batch ])
    v_batch    = np.vstack([ b.value  for b in batch ])
    x2_batch   = np.vstack([ b.x2     for b in batch ])

    sess.run(policy.optim, feed_dict={ policy.x     : x_batch,
                                       policy.uref  : u_batch })
    sess.run(state .optim, feed_dict={ state .x     : x_batch,
                                       state .x2ref : x2_batch })
    sess.run(value.optim,  feed_dict={ value.x      : x_batch,
                                       value.vref   : v_batch })


U  = sess.run(policy.policy, feed_dict={ policy.x: env.obs(X.T).T })
V  = sess.run(value.value,   feed_dict={ value.x:  env.obs(X.T).T })
X2 = sess.run(state.x2,      feed_dict={ state.x2: env.obs(X.T).T })

# Scatter plot of policy/value funciton sampling (in file)
plt.clf()
plt.subplot(1,3,1)
plt.scatter(X[:,0].flat,X[:,1].flat,c=U[:,0],s=50,linewidths=0,alpha=.8,vmin=-2,vmax=2)
plt.colorbar()
plt.subplot(1,3,2)
plt.scatter(X[:,0].flat,X[:,1].flat,c=U[:,1],s=50,linewidths=0,alpha=.8,vmin=-2,vmax=2)
plt.colorbar()
plt.subplot(1,3,3)
plt.scatter(X[:,0].flat,X[:,1].flat,c=V[:],s=50,linewidths=0,alpha=.8)
plt.colorbar()
plt.show()

'''
for episode in range(NEPISODES):
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
    sess.run(policyTarget.update_variables)
    sess.run(qvalueTarget.update_variables)

    # Display and logging (not mandatory).
    maxq = np.max( sess.run(qvalue.qvalue,feed_dict={ qvalue.x : x_batch,
                                                      qvalue.u : u_batch }) ) \
                                                      if 'x_batch' in locals() else 0
    print 'Ep#{:3d}: max qvalue={:2.3f}' \
        .format(episode, maxq)
    h_qva.append(maxq)
    h_ste.append(step)

    if RENDERRATE and not (episode+1) % RENDERRATE:     
        if 'saveweights' in RENDERACTION:
            tf.train.Saver().save(sess,'netvalues/double/actorcritic_double.%04d' % episode)

        if 'rollout' in RENDERACTION:
            # Rollout and render in Gepetto
            rendertrial(100)

        if 'draw' in RENDERACTION:
            # Generate sampling of the policy/value functions

            U = sess.run(policy.policy, feed_dict={ policy.x: env.obs(X.T).T })
            Q = sess.run(qvalue.qvalue, feed_dict={ qvalue.x: env.obs(X.T).T,
                                                    qvalue.u: U })
            # Scatter plot of policy/value funciton sampling (in file)
            plt.clf()
            plt.subplot(1,2,1)
            plt.scatter(X[:,0].flat,X[:,1].flat,c=U[:],s=50,linewidths=0,alpha=.8,vmin=-2,vmax=2)
            plt.colorbar()
            plt.subplot(1,2,2)
            plt.scatter(X[:,0].flat,X[:,1].flat,c=Q[:],s=50,linewidths=0,alpha=.8)
            plt.colorbar()
            plt.savefig('figs/actorcritic_%04d.png' % episode)

# \\\END_FOR episode in range(NEPISODES)
'''
