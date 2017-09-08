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

for d in data:
    d.U[-1,:] = d.U[-2,:]

#nex = lambda X: np.vstack([ X[1:,:], X[-1:,:]])          # Shift once 
SHIFT = 4
nex = lambda X: np.vstack([ X[SHIFT:,:] ] +  [ X[-1:,:] ]*SHIFT)  # shit 5 times
xs =   [ d.X        for d in data ]; xs   = np.array(np.vstack(xs))
us =   [ d.U        for d in data ]; us   = np.array(np.vstack(us))
nexs = [ nex(d.X)   for d in data ]; nexs = np.array(np.vstack(nexs))
vs =   [ d.cost-d.T for d in data ]; vs   = np.expand_dims(np.array(np.concatenate(vs)),1)

subtraj = lambda X,i:    np.ravel( np.vstack([X[i:,:]]+[X[-1:,:]]*i) )
xplust = lambda X,T: np.hstack([ X, np.expand_dims(T[-1]-T,1) ])
#trajs = [ subtraj(xplust(d.X,d.T),i) for d in data for i in range(len(d.X))  ]; trajs = np.array(np.vstack(trajs))
trajxs = [ subtraj(d.X,i) for d in data for i in range(len(d.X))  ]; trajxs = np.array(np.vstack(trajxs))
trajus = [ subtraj(d.U,i) for d in data for i in range(len(d.U))  ]; trajus = np.array(np.vstack(trajus))


# --- NETS ---------------------------------------------------------
# --- NETS ---------------------------------------------------------
# --- NETS ---------------------------------------------------------
from networks import *

policy = PolicyNetwork(NX,NU,umax=[-1,25]).setupOptim('direct')
pstate = PolicyNetwork(NX,NX             ).setupOptim('direct')
ptrajx = PolicyNetwork(NX,trajxs.shape[1] ).setupOptim('direct')
ptraju = PolicyNetwork(NX,trajus.shape[1], umax=[-1,25] ).setupOptim('direct')
value  = PolicyNetwork(NX,1              ).setupOptim('direct')

sess            = tf.InteractiveSession()
tf.global_variables_initializer().run()

#tf.train.Saver().restore(sess,dataRootPath+'/policy')

# --- TRAINING -----------------------------------------------------
BATCH_SIZE = 128
#NEPISODES  = 50000

def train(nets = None,NEPISODES=10000,track=False):
    if nets is None:
        nets = [
                policy,
                pstate,
                ptrajx,
                ptraju,
                value,
                ]
    REFBATCH = random.sample(range(len(us)),BATCH_SIZE*8)
    hist = []
    for episode in range(NEPISODES):
        if not episode % 100: 
            print 'Episode #',episode

        batch = random.sample(range(len(us)),BATCH_SIZE)
        # sess.run(policy.optim, feed_dict={ policy.x     : xs[batch,:],
        #                                    policy.uref  : us[batch,:] })
        # sess.run(pstate.optim, feed_dict={ pstate.x     : xs[batch,:],
        #                                    pstate.uref  : nexs[batch,:] })
        # sess.run(ptrajx.optim, feed_dict={ ptrajx .x     : xs[batch,:],
        #                                    ptrajx .uref  : trajxs[batch,:] })
        # sess.run(ptraju.optim, feed_dict={ ptraju .x     : xs[batch,:],
        #                                    ptraju .uref  : trajus[batch,:] })
        # sess.run(value .optim, feed_dict={ value .x     : xs[batch,:],
        #                                    value .uref  : vs[batch,:] })

        sess.run([ p.optim for p in nets],
                 feed_dict={ policy.x      : xs    [batch,:],
                             policy.uref   : us    [batch,:] ,
                             pstate.x      : xs    [batch,:],
                             pstate.uref   : nexs  [batch,:],
                             ptrajx .x     : xs    [batch,:],
                             ptrajx .uref  : trajxs[batch,:],
                             ptraju .x     : xs    [batch,:],
                             ptraju .uref  : trajus[batch,:],
                             value .x      : xs    [batch,:],
                             value .uref   : vs    [batch,:] })
        if track and not episode % 10:
            U = sess.run(policy.policy, feed_dict={ policy.x     : xs[REFBATCH,:] })
            hist.append( norm(U-us[REFBATCH])/len(REFBATCH) )

    return hist

hist = train(NEPISODES=int(1e5))

# --- SAVE ---------------------------------------------------------
#tf.train.Saver().save(sess,dataRootPath+'/policy')

# --- ROLLOUT ------------------------------------------------------

def trajFromU(x0 = None, NSTEPS = 300,THRESHOLD = 8e-2,withPlot=None,**plotargs):
    x = x0 if x0 is not None else env.sample()
    hx =  [ x.T ]
    hu =  []
    for i in range(NSTEPS):
        u = sess.run(policy.policy, feed_dict={ policy.x: x.T } ).T
        x = env.dynamics(x,np.matrix(u))
        if norm(x) < THRESHOLD: break
        hx.append(x.T)
        hu.append(u.T)
    X = np.vstack(hx)
    U = np.vstack( hu + [u.T] )
    T = i*env.DT
    if withPlot is not None: plt.plot(X[:,0],X[:,1],withPlot,**plotargs)
    return X,U,T

def trajFromX(x0 = None, NSTEPS = 10,THRESHOLD = 8e-2,withPlot=None,**plotargs):
    x = x0.T if x0 is not None else env.sample().T
    hx =  [ x ]
    for i in range(NSTEPS):
        x = sess.run(pstate.policy, feed_dict={ pstate.x: x } )
        if norm(x) < THRESHOLD: break
        hx.append(x)
    X = np.vstack(hx)
    if withPlot is not None: plt.plot(X[:,0],X[:,1],withPlot,**plotargs)
    return X

def trajFromTraj(x0 = None,withPlot=None,**plotargs):
    x = x0.T if x0 is not None else env.sample().T
    X = sess.run(ptrajx.policy,feed_dict={ ptrajx.x:x })
    X = np.reshape(X,[max(X.shape)/NX,NX])
    U = sess.run(ptraju.policy,feed_dict={ ptraju.x:x })
    U = np.reshape(U,[max(U.shape)/NU,NU])
    if withPlot is not None: plt.plot(X[:,0],X[:,1],withPlot,**plotargs)
    return X,U,0

def sampleTrajs(x0,DNEIGHBORGH=3e-1,withPlot = None, **plotargs):
    res = [ d.X for i,d in enumerate(data) if norm(d.x0-x0)<DNEIGHBORGH ]
    if withPlot is not None:
        for X in res: 
            plt.plot(X[:,0],X[:,1],withPlot,**plotargs)



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
env.vlow   *=   0
env.vup    *=   0

def trial(x0 = None,dneigh=3e-1):
    x0 = x0 if x0 is not None else env.sample()
    sampleTrajs(x0,DNEIGHBORGH=dneigh,withPlot='k')
    xu = trajFromU(x0,withPlot='r+-',linewidth=3)
    xx = trajFromX(x0,withPlot='b+-',linewidth=3)
    xt = trajFromTraj(x0,withPlot='g+-',linewidth=3)

#x = env.reset()
#x[NQ:] = 0
# def trial(x0 = None,dneigh=3e-1):
#     NSTEPS = 300
#     if x0 is None:
#         x0 = env.reset()
#         x0[NQ:] = 0
#         #x = np.matrix([ 1.,0,0, 0,0,0 ]).T

#     hx = zero([NSTEPS,NX])
#     hy = zero([NSTEPS,NX])
#     x = x0.copy()
#     y = x0.copy().T
#     print x.T
#     for i in range(NSTEPS):
#         hx[i,:] = x.T
#         hy[i,:] = y
#         u = sess.run(policy.policy, feed_dict={ policy.x: x.T } )
#         x = env.dynamics(x,np.matrix(u.T))
#         y = sess.run(pstate.policy,feed_dict={ pstate.x: y })

#     for i in [ i for i,d in enumerate(data) if norm(d.x0-x0)<dneigh ]:
#         plt.plot(data[i].X[:,0],data[i].X[:,1],'k')
#     plt.plot(hx[:,0],hx[:,1],'r+-',linewidth=2)
#     plt.plot(hy[:,0],hy[:,1],'b+-',linewidth=2)

#     X = sess.run(ptraj.policy,feed_dict={ ptraj.x:x0.T })
#     plt.plot(X[0,::6],X[0,1::6],'g',linewidth=5)

# x0 = zero(6)
# x  = x0.T
# hx = []
# for i in range(50): 
#     hx.append(x)
#     x = sess.run(pstate.policy,feed_dict={ pstate.x: x })


# --- OPTIMIZE -----------------------------------------------------
'''
x0 = env.sample()

X,U,T = trajFromU(x0)
#T = sess.run( value.policy, feed_dict = { value.state: x0.T })

from specpath import acadoBinDir,acadoTxtPath
from acado_connect import AcadoConnect

acado = AcadoConnect(acadoBinDir+"connect_bicopter",
                     datadir=acadoTxtPath)
acado.NQ = NQ
acado.NV = NV

acado.options['thetamax']    = np.pi/2
acado.options['printlevel']    = 1
if 'shift' in acado.options: del acado.options['shift']
if env is not None:
    acado.options['umax']     = "%.2f %.2f" % tuple([x for x in env.umax])
          
#if 'icontrol' in acado.options: del acado.options['icontrol']
acado.debug(False)
acado.iter                = 20
acado.options['steps']    = 20

ts = np.arange(0.,1.,1./X.shape[0])
np.savetxt(acado.stateFile  ('i'),  np.vstack([ ts, X.T]).T )
np.savetxt(acado.controlFile('i'),  np.vstack([ ts, U.T]).T )

acado.setTimeInterval(T)
acado.run(x0,zero(6),autoInit=False)

Xa = acado.states()
Ua = acado.controls()

plt.plot(X [:,0],X [:,1],'g')
plt.plot(Xa[:,0],Xa[:,1],'r')
'''
