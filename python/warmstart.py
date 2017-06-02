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
NEPISODES               = 500           # Max training steps
NSTEPS                  = 100           # Max episode length
QVALUE_LEARNING_RATE    = 0.001         # Base learning rate for the Q-value Network
POLICY_LEARNING_RATE    = 0.0001        # Base learning rate for the policy network
DECAY_RATE              = 0.99          # Discount factor 
UPDATE_RATE             = 0.01          # Homotopy rate to update the networks
REPLAY_SIZE             = 10000         # Size of replay buffer
BATCH_SIZE              = 64            # Number of points to be fed in stochastic gradient
NH1 = NH2               = 250           # Hidden layer size
RESTORE                 = "netvalues/actorcritic.15.kf2" # Previously optimize net weight 
                                        # (set empty string if no)
### --- Environment
env                 = Pendulum(1)       # Continuous pendulum
env.withSinCos      = True              # State is dim-3: (cosq,sinq,qdot) ...
NX                  = env.nobs          # ... training converges with q,qdot with 2x more neurones.
NU                  = env.nu            # Control is dim-1: joint torque

env.vmax            = 100.
env.DT              = .15
env.NDT             = 2
env.Kf              = 0.2
NSTEPS              = 30

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

def rendertrial(x0=None,maxiter=NSTEPS,verbose=True):
    x = env.reset(x0)
    rsum = 0.
    for i in range(maxiter):
        u = sess.run(policy.policy, feed_dict={ policy.x: x.T })
        x, reward = env.step(u)
        env.render()
        time.sleep(1e-2)
        rsum += reward
    if verbose: print 'Lasted ',i,' timestep -- total reward:',rsum
signal.signal(signal.SIGTSTP, lambda x,y:rendertrial()) # Roll-out when CTRL-Z is pressed

env.NDT = 1
env.modulo = False

from acado_runner import AcadoRunner
acado = AcadoRunner()
acado.options['horizon']  = NSTEPS*env.DT
acado.options['steps']    = NSTEPS
acado.options['shift']    = 0
acado.options['iter']     = 20
acado.options['friction'] = env.Kf
acado.options['decay'] = DECAY_RATE
acado.options['icontrol']    = '/tmp/guess.ctl'
del acado.options['istate']

def policyOptim(x0):
    '''
    Rollout the net to get a warmstart, then run Acado to optimize the warmstart
    Return the policy and the value at input state.
    '''
    env.reset(x0)
    fctl = open(acado.options['icontrol'],'w')
    for i in range(NSTEPS+1):
        u = sess.run(policy.policy, feed_dict={ policy.x: env.obs(env.x).T })
        env.step(u)
        fctl.write( '%.10f \t%.20f\n' % ( i*env.DT, u[0,0] ) )
    fctl.close()
    mod = round(env.x[0,0]/2/np.pi)
    x0[0,0] -= mod*2*np.pi
    return acado.run(x0[0,0],x0[1,0])

#x0 = np.matrix([3.0,0.0]).T
#acado.options['iter']     = 0
#acado.additionalOptions = ' --plot '
#policyOptim(x0)

def explore(NROLLOUTS=1000):
    '''
    Run NROLLOUTS computations of the optimal policy (obtained from net
    warmstart and acado optim), save the dataset in databasexx_DATA.np and
    returns it.
    '''
    database = []
    for rollout in range(NROLLOUTS):
        if not rollout % 100: print "=== Rollout #",rollout
        env.reset()
        x = env.x.copy()
        u,cost = policyOptim(env.x)
        database.append( [ x[0,0], x[1,0], u[0], cost ] )
    
    D=np.array(database)
    import datetime
    filename = 'databasexx_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S')+'.np'
    np.save(open(filename,'w'),D)

    return D

#acado.additionalOptions = ' --plot '
def rollout(x0 = None, interactive = False):
    '''Execute one optimal runout using the optimized policy (net+acado) from random init point.'''
    env.reset(x0)
    print env.x.T
    for i in range(NSTEPS):
        x = env.x.copy()
        u,_ = policyOptim(x)
        env.reset(x)
        env.step(np.matrix(u))
        env.render()
        if interactive: 
            print acado.cmd
            raw_input("Press Enter to continue...")


acadompc = AcadoRunner()
acadompc.options['horizon']  = NSTEPS*env.DT
acadompc.options['steps']    = NSTEPS
acadompc.options['friction'] = env.Kf
acadompc.options['decay'] = DECAY_RATE
del acadompc.options['istate']

def rolloutmpc(x0 = None, interactive = False):
    env.reset(x0)
    print env.x.T
    u,_ = policyOptim(env.x)  # init guess
    env.step(np.matrix(u))
    env.render()
    print acado.cmd
    raw_input("Press Enter to continue...")
    for i in range(NSTEPS):
        u,cost = acadompc.run(env.x[0,0],env.x[1,0])
        env.step(np.matrix(u))
        env.render()
        if interactive: 
            print acadompc.cmd
            raw_input("Press Enter to continue...")



