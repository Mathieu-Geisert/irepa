from pinocchio.utils import *
from numpy.linalg import inv,norm
import math
import time
import random
import matplotlib.pylab as plt
plt.ion()

# --- ENV ----------------------------------------------------------
# --- ENV ----------------------------------------------------------
# --- ENV ----------------------------------------------------------
from bicopter import Bicopter

env     = Bicopter(withDisplay=False)
NX      = 6
NQ      = 3
NV      = 3
NU      = 2
UBOUND  = [ env.umin[0], env.umax[0] ]

# --- DATA ---------------------------------------------------------
# --- DATA ---------------------------------------------------------
# --- DATA ---------------------------------------------------------
from grid_policy import GridPolicy,onehot
dataRootPath = 'data/planner/bicopter'

grid = GridPolicy()
grid.load(dataRootPath+'/grid_sampled.npy')

data = [ d for d in grid.data if len(d.X)>0 and abs(d.x0[2,0])<.7]

# Data set with only starting positions
'''
xs   = [ d.x0.T   for d in data  ]; xs   = np.array(np.vstack(xs))
us   = [ d.U[0,:] for d in data  ]; us   = np.array(np.vstack(us))
nexs = [ d.X[1,:] for d in data  ]; nexs = np.array(np.vstack(nexs))
vs   = [ d.cost   for d in data  ]; vs   = np.array(np.vstack(vs))
'''

nex = lambda X: np.vstack([ X[1:,:], X[-1:,:]])

xs =   [ d.X        for d in data ]; xs   = np.array(np.vstack(xs))
us =   [ d.U        for d in data ]; us   = np.array(np.vstack(us))
nexs = [ nex(d.X)   for d in data ]; nexs = np.array(np.vstack(nexs))
vs =   [ d.cost-d.T for d in data ]; vs   = np.array(np.vstack(vs))

# --- NETS ---------------------------------------------------------
# --- NETS ---------------------------------------------------------
# --- NETS ---------------------------------------------------------
from networks import *

policy = PolicyNetwork(NX,NU,umax=[-1,25]).setupOptim('direct')
value  = PolicyNetwork(NX,1)         .setupOptim('direct')

sess            = tf.InteractiveSession()
tf.global_variables_initializer().run()

# --- TRAINING -----------------------------------------------------
BATCH_SIZE = 128
NEPISODES  = 10000

REFBATCH = random.sample(range(len(us)),BATCH_SIZE*8)

hist = []
for episode in range(NEPISODES):
    if not episode % 100: 
        print 'Episode #',episode

    batch = random.sample(range(len(us)),BATCH_SIZE)

    sess.run(policy.optim, feed_dict={ policy.x     : xs[batch,:],
                                       policy.uref  : us[batch,:] })

    if not episode % 10:
        U = sess.run(policy.policy, feed_dict={ policy.x     : xs[REFBATCH,:] })
        hist.append( norm(U-us[REFBATCH])/len(REFBATCH) )

# --- SAVE ---------------------------------------------------------
#tf.train.Saver().save(sess,dataRootPath+'/policy')


# --- PLAY AND DEBUG -----------------------------------------------

xaxis = onehot(0,6)
yaxis = onehot(1,6)
caxis = onehot(2,6)

ds = grid.dataOnAPlane(xaxis,yaxis,caxis*0,eps=3e-2)
D  = grid.np(ds)

U  = sess.run(policy.policy, feed_dict={ policy.x  : D[:,:NX] })

'''
plt.subplot(2,2,1)
plt.scatter((D[:,:NX]*xaxis).flat,(D[:,:NX]*yaxis).flat,c=D[:,-3].flat,s=200,linewidths=0)
plt.title('Uref 0')
plt.subplot(2,2,2)
plt.scatter((D[:,:NX]*xaxis).flat,(D[:,:NX]*yaxis).flat,c=D[:,-2].flat,s=200,linewidths=0)
plt.title('Uref 1')
plt.subplot(2,2,3)
plt.scatter((D[:,:NX]*xaxis).flat,(D[:,:NX]*yaxis).flat,c=U[:,0].flat,s=200,linewidths=0)
plt.title('Ulearn 0')
plt.subplot(2,2,4)
plt.scatter((D[:,:NX]*xaxis).flat,(D[:,:NX]*yaxis).flat,c=U[:,1].flat,s=200,linewidths=0)
plt.title('Ulearn 1')
'''


env.qlow[2] = -.7
env.qup [2] = +.7

#x = env.reset()
#x[NQ:] = 0
def trial(x0 = None,dneigh=3e-1):
    NSTEPS = 300
    if x0 is None:
        x0 = env.reset()
        x0[NQ:] = 0
        #x = np.matrix([ 1.,0,0, 0,0,0 ]).T

    hx = zero([NSTEPS,NX])
    x = x0.copy()
    print x.T
    for i in range(300):
        u = sess.run(policy.policy, feed_dict={ policy.x: x.T } )
        x = env.dynamics(x,np.matrix(u.T))
        hx[i,:] = x.T

    plt.plot(hx[:,0],hx[:,1],'r',linewidth=2)
    for i in [ i for i,d in enumerate(data) if norm(d.x0-x0)<dneigh ]:
        plt.plot(data[i].X[:,0],data[i].X[:,1],'k')

