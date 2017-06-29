'''
Use a trained actor-critic net to warm start a continous 2nd order acado using only shooting states.
The warm start is quite sparse, as only some few shooting nodes are initialized. 
The script might be used to generate dataset of optimal rollouts, to be used to train a policy (not done yet).

TODO: 
- generate valid dataset, check the dataset, possibly reduce the number of shooting nodes. 
- train the policy net from the dataset. It should predict next shooting node from current node.
- Warm start Acado from the second net.
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
from pinocchio.utils import zero,rand

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
NEPISODES               = 500           # Max training steps
NSTEPS                  = 100           # Max episode length
QVALUE_LEARNING_RATE    = 0.001         # Base learning rate for the Q-value Network
POLICY_LEARNING_RATE    = 0.0001        # Base learning rate for the policy network
DECAY_RATE              = 0.99          # Discount factor 
UPDATE_RATE             = 0.01          # Homotopy rate to update the networks
REPLAY_SIZE             = 10000         # Size of replay buffer
BATCH_SIZE              = 64            # Number of points to be fed in stochastic gradient
NH1 = NH2               = 250           # Hidden layer size
RESTORE                 = "netvalues/actorcritic.dt015.kf02.ep1300" # Previously optimize net weight 
                                        # (set empty string if no)
### --- Environment
env                 = Pendulum(1)       # Continuous pendulum
env.withSinCos      = True              # State is dim-3: (cosq,sinq,qdot) ...
NX                  = env.nobs          # ... training converges with q,qdot with 2x more neurones.
NU                  = env.nu            # Control is dim-1: joint torque

env.vmax            = 100.
env.Kf              = 0.2
env.modulo          = False

env.DT              = 0.15
env.NDT             = 1
NSTEPS              = 32                # Number of intergration steps in horizon
NNODES              = 8                 # Number of shooting nodes
FNODES              = NSTEPS/NNODES     # Number of integration nodes per shooting interval ...
assert(not NSTEPS % NNODES)             # ... should be an integer

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


### --- Tensor flow initialization

policy          = PolicyNetwork(). setupOptim()
policyTarget    = PolicyNetwork(). setupTargetAssign(policy)

qvalue          = QValueNetwork(). setupOptim()
qvalueTarget    = QValueNetwork(). setupTargetAssign(qvalue)

sess            = tf.InteractiveSession()
tf.global_variables_initializer().run()

if len(RESTORE)>0:
    tf.train.Saver().restore(sess, RESTORE)

def rollout(x0=None,display=False,tsleep=0):
    '''Rollout a trajectory from the AC network. Return the X and U trajectories.'''
    hx = []
    hu = []
    env.reset(x0)
    for i in range(NSTEPS+1):
        u = sess.run(policy.policy, feed_dict={ policy.x: env.obs(env.x).T })
        hx.append(env.x.copy().T)
        hu.append(u.copy())
        env.step(u)
        if display:
            env.render()
            time.sleep(tsleep)

    X = np.vstack(hx)
    U = np.vstack(hu)

    mod = round(env.x[0,0]/2/np.pi)
    dx = -mod*2*np.pi
    X[:,0] += dx
    return X,U

x0=np.matrix([ 2., .2 ]).T
#X,U=rollout(x0)

env.NDT = 10
x0 = np.matrix([ 0.30459664, -7.20891078]).T
NNODES = 16
FNODES = NSTEPS/NNODES

from acado_runner import AcadoRunner
acado = AcadoRunner("/home/nmansard/src/pinocchio/pycado/build/unittest/pendulum2o")
acado.options['horizon']  = NSTEPS*env.DT
acado.options['steps']    = NNODES
acado.options['shift']    = 0
acado.options['iter']     = 10 # 100
acado.options['friction'] = env.Kf
#acado.options['decay']    = 0 # DECAY_RATE
#del acado.options['istate']

def policyOptim(x0=None):
    X,U = rollout(x0)

    guess = np.hstack([ np.matrix(np.arange(0,(NSTEPS+1)*env.DT,env.DT)).T,
                        X, U, zero(NSTEPS+1), zero(NSTEPS+1) ])[::FNODES]

    np.savetxt('/tmp/state.txt',guess)
    acado.options['istate']='/tmp/state.txt'
    if 'icontrol' in acado.options: 
        del acado.options['icontrol']
    
    u,cost = acado.run(X[0,:1],X[0,1:])
    return u,cost,X[0,:]
    

#acado.debug()



policyOptim(x0)
# for i in range(10):
#     policyOptim()

def explore(nrollout):
    acado.debug(False)  # just in case
    DX      = []
    DU      = []
    idxHead = [] # Only indices of trajectories head are kept here
    Xreject = []
    X0      = []

    for rollouts in range(nrollout):
        if not rollouts % 100: print "# Rollout ", rollouts
        try:
            env.reset()
            x0 = env.x.copy()
            policyOptim(x0)
        except: 
            Xreject.append( x0.T )
            continue
        X0.append(x0.T)
        X = acado.states()[:,:2]
        U = acado.states()[:,2:3]
        idxHead.append(len(DX))
        for i in range(X.shape[0]-1):
            DX.append( np.hstack([ X[i,:], X[i+1,:] ]) )
            DU.append( np.hstack([ X[i,:], U[i,:]   ]) )

    return np.vstack(DX),np.vstack(DU),idxHead, np.vstack(Xreject),np.vstack(X0)
    

for i in range(00):
    print " *** Exploration #",i
    DX,DU,idx,Xfail,X0=explore(1000)
    np.save('DX%04d'%i,DX)
    np.save('DU%04d'%i,DX)
    np.save('idx%04d'%i,idx)
    np.save('Xfail%04d'%i,Xfail)
    np.save('Xinit%04d'%i,X0)

if 'DX' not in locals():
    DX = []
    DU = []
    idx = []
    Xfail = []
    nbsamples=0
    X0 = []
    for i in range(25):
        print 'Load %04d'%i
        DX.append(np.load('DX%04d.npy'%i))
        DU.append(np.load('DU%04d.npy'%i))
        idx.append(np.load('idx%04d.npy'%i)+nbsamples)
        Xfail.append(np.load('Xfail%04d.npy'%i))
        X0.append(np.load('Xinit%04d.npy'%i))
        nbsamples += DX[-1].shape[0]
    DX = np.vstack(DX)
    DU = np.vstack(DU)
    idx = np.hstack(idx)
    Xfail = np.vstack(Xfail)
    X0 = np.vstack(X0)

#plt.scatter(DX[:,0],DX[:,1],c=DX[:,2],s=50,linewidths=0,alpha=.8)
#plt.scatter(Xfail[:,0].flat,Xfail[:,1].flat,c='k',s=5,alpha=.2)

plt.ion()
plt.scatter(DX[idx,0],DX[idx,1],c=DX[idx,2],s=50,linewidths=0,alpha=.8)
plt.scatter(Xfail[:,0].flat,Xfail[:,1].flat,c='k',s=15,alpha=.25)

# itrue   = 34784
# ifalse  = 2372
# xtrue   = np.matrix(DX[itrue,:2]).T
# xfalse  = np.matrix(Xfail[ifalse,:]).T

# --- Test with discrete pendulum  ... not working ?WHY?
# acadoD = AcadoRunner()
# acadoD.options['horizon']  = NSTEPS*env.DT
# acadoD.options['steps']    = NSTEPS
# acadoD.options['shift']    = 0
# acadoD.options['iter']     = 100
# acadoD.options['friction'] = env.Kf

# np.savetxt('/tmp/stateD.txt', 
#            np.hstack([ np.matrix(np.arange(0,(NSTEPS+1)*env.DT,env.DT)).T,
#                        X, zero(NSTEPS+1) ]))
# np.savetxt('/tmp/controlD.txt', 
#            np.hstack([ np.matrix(np.arange(0,(NSTEPS+1)*env.DT,env.DT)).T,
#                        U ]))

# acadoD.options['istate'  ]='/tmp/stateD.txt'
# acadoD.options['icontrol']='/tmp/controlD.txt'
# acadoD.debug()
# u,cost = acadoD.run(x0[0:1,0],x0[1:2,0])


