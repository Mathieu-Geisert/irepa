from pinocchio.utils import *
from numpy.linalg import inv,norm
import math
import time
import random
import matplotlib.pylab as plt
from collections import namedtuple
import pylab
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


class Dataset:
    def __init__(self,datafile=dataRootPath+'/grid.npy'):
        self.grid = GridPolicy()
        self.grid.load(datafile)
        for d in self.grid.data:  # Correct error at the end of U traj
            d.U[-1,:] = d.U[-2,:]

        # For quad copter: remove terminal traj
        self.data = [ d for d in self.grid.data if len(d.X)>0 and abs(d.x0[2,0])<.7]

    def set(self):
        data = self.data
        SHIFT = 4
        nex = lambda X: np.vstack([ X[SHIFT:,:] ] +  [ X[-1:,:] ]*SHIFT)  # shit 5 times

        xs =   [ d.X        for d in data ]
        self.xs   = np.array(np.vstack(xs))

        us =   [ d.U        for d in data ]
        self.us   = np.array(np.vstack(us))

        nexs = [ nex(d.X)   for d in data ]
        self.nexs = np.array(np.vstack(nexs))

        vs =   [ d.cost-d.T for d in data ]
        self.vs   = np.expand_dims(np.array(np.concatenate(vs)),1)
        
        subtraj = lambda X,i:    np.ravel( np.vstack([X[i:,:]]+[X[-1:,:]]*i) )
        xplust = lambda X,T: np.hstack([ X, np.expand_dims(T[-1]-T,1) ])

        trajxs = [ subtraj(d.X,i) for d in data for i in range(len(d.X))  ]
        self.trajxs = np.array(np.vstack(trajxs))

        trajus = [ subtraj(d.U,i) for d in data for i in range(len(d.U))  ]
        self.trajus = np.array(np.vstack(trajus))
        
        return self


# --- NETS ---------------------------------------------------------
# --- NETS ---------------------------------------------------------
# --- NETS ---------------------------------------------------------
from networks import *

class Nets:
    BATCH_SIZE = 128

    def __init__(self, trajlength):
        self.policy = PolicyNetwork(NX,NU,umax=[-1,25]).setupOptim('direct')

        bx     = [ 5.,5.,np.pi/2, 10.,10.,10. ]
        self.pstate = PolicyNetwork(NX,NX,umax=[ [-x for x in bx],bx ]  ).setupOptim('direct')

        bx     = bx*trajlength
        bx     = [ [ -x for x in bx ], bx ]
        self.ptrajx = PolicyNetwork(NX,trajlength*NX, umax=bx      ).setupOptim('direct')
        self.ptraju = PolicyNetwork(NX,trajlength*NU, umax=[-1,25] ).setupOptim('direct')

        self.value  = PolicyNetwork(NX,1).setupOptim('direct')

        self.sess   = tf.InteractiveSession()
        tf.global_variables_initializer().run()

    def load(self):
        tf.train.Saver().restore(self.sess,dataRootPath+'/policy')
    def save(self):
        tf.train.Saver().save   (self.sess,dataRootPath+'/policy')

    # --- TRAINING -----------------------------------------------------
    def train(self,dataset,nets = None,NEPISODES=10000,track=False):
        if nets is None:
            nets = [
                self.policy,
                self.pstate,
                self.ptrajx,
                self.ptraju,
                self.value,
                ]

        if track: 
            REFBATCH = random.sample(range(len(dataset.us)),self.BATCH_SIZE*8)
        hist = []

        for episode in range(NEPISODES):
            if not episode % 100: 
                print 'Episode #',episode

            batch = random.sample(range(len(dataset.us)),self.BATCH_SIZE)
            self.sess.run([ p.optim for p in nets],
                          feed_dict={ self.policy.x      : dataset.xs    [batch,:],
                                      self.policy.uref   : dataset.us    [batch,:] ,
                                      self.pstate.x      : dataset.xs    [batch,:],
                                      self.pstate.uref   : dataset.nexs  [batch,:],
                                      self.ptrajx .x     : dataset.xs    [batch,:],
                                      self.ptrajx .uref  : dataset.trajxs[batch,:],
                                      self.ptraju .x     : dataset.xs    [batch,:],
                                      self. ptraju.uref  : dataset.trajus[batch,:],
                                      self.value .x      : dataset.xs    [batch,:],
                                          self.value .uref   : dataset.vs    [batch,:] })
            
            if track and not episode % 10:
                U = self.sess.run( self.policy.policy,
                                   feed_dict={ self.policy.x     : dataset.xs[REFBATCH,:] })
                hist.append( norm(U-dataset.us[REFBATCH])/len(REFBATCH) )

        return hist


# --- ROLLOUT ------------------------------------------------------
# --- ROLLOUT ------------------------------------------------------
# --- ROLLOUT ------------------------------------------------------

def trajFromU(x0 = None, NSTEPS = 300,THRESHOLD = 8e-2,withPlot=None,**plotargs):
    x = x0 if x0 is not None else env.sample()
    hx =  [ x.T ]
    hu =  []
    for i in range(NSTEPS):
        u = nets.sess.run(nets.policy.policy, feed_dict={ nets.policy.x: x.T } ).T
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
    hx = [ x ]
    hu = [ ]
    for i in range(NSTEPS):
        x,u = nets.sess.run([nets.pstate.policy,nets.policy.policy], 
                            feed_dict={ nets.pstate.x: x,
                                        nets.policy.x: x } )
        if norm(x) < THRESHOLD: break
        hx.append(x)
        hu.append(u)
    X = np.vstack(hx)
    U = np.vstack(hu+[u])
    T = nets.sess.run(nets.value.policy,feed_dict={ nets.value.x:x })[0,0]
    if withPlot is not None: plt.plot(X[:,0],X[:,1],withPlot,**plotargs)
    return X,U,T

def trajFromTraj(x0 = None,withPlot=None,**plotargs):
    x = x0.T if x0 is not None else env.sample().T
    X = nets.sess.run(nets.ptrajx.policy,feed_dict={ nets.ptrajx.x:x })
    X = np.reshape(X,[max(X.shape)/NX,NX])
    U = nets.sess.run(nets.ptraju.policy,feed_dict={ nets.ptraju.x:x })
    U = np.reshape(U,[max(U.shape)/NU,NU])
    T = nets.sess.run(nets.value.policy,feed_dict={ nets.value.x:x })[0,0]
    if withPlot is not None: plt.plot(X[:,0],X[:,1],withPlot,**plotargs)
    return X,U,T

def sampleTrajs(x0,DNEIGHBORGH=3e-1,withPlot = None, **plotargs):
    '''Get a bundle of trajectories in the dataset, whose init point are close to x0.'''
    res = [ d.X for i,d in enumerate(dataset.data) if norm(d.x0-x0)<DNEIGHBORGH ]
    if withPlot is not None:
        for X in res: 
            plt.plot(X[:,0],X[:,1],withPlot,**plotargs)




# --- OPTIMIZE -----------------------------------------------------

# --- SETUP ACADO
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
acado.iter                = 80
acado.options['steps']    = 20


# --- NET+ACADO

def optNet(x0,net=trajFromU,withPlot=False,color='r',**plotargs):
    X,U,T = net(x0)
    if withPlot:
        plt.plot(X [:,0],X [:,1],'--',color=color,**plotargs)

    #ts = np.arange(0.,1.,1./X.shape[0])
    ts = np.arange(0.,X.shape[0])/(X.shape[0]-1)

    np.savetxt(acado.stateFile  ('i'),  np.vstack([ ts, X.T]).T )
    np.savetxt(acado.controlFile('i'),  np.vstack([ ts, U.T]).T )
    acado.setTimeInterval(T)
    acado.options['Tmax'] = T*10
    if not acado.run(x0,zero(6),autoInit=False):
        raise RuntimeError("Error: Acado returned false")

    Xa = acado.states()
    Ua = acado.controls()
    Ta = acado.opttime()

    if withPlot:
        plt.plot(Xa[:,0],Xa[:,1],'-', color=color,**plotargs)
        
    return Xa,Ua,Ta

# --- PLAY AND DEBUG -----------------------------------------------
# --- PLAY AND DEBUG -----------------------------------------------
# --- PLAY AND DEBUG -----------------------------------------------

env.qlow[2] = -.7
env.qup [2] = +.7
env.vlow   *=   0
env.vup    *=   0

def trial(x0 = None,dneigh=1.2e-1,opt=True):
    x0 = x0 if x0 is not None else env.sample()
    print x0.T
    sampleTrajs(x0,DNEIGHBORGH=dneigh,withPlot='k')
    if opt:
        try: optNet(x0,net=trajFromU,withPlot=True,color='r',linewidth=2.5)
        except:pass
        try: optNet(x0,net=trajFromX,withPlot=True,color='b',linewidth=2.5)
        except:pass
        try: optNet(x0,net=trajFromTraj,withPlot=True,color='y',linewidth=2.5)
        except:pass
    else:
        xu = trajFromU(x0,withPlot='r+-',linewidth=2.5)
        xx = trajFromX(x0,withPlot='b+-',linewidth=2.5)
        xt = trajFromTraj(x0,withPlot='y+-',linewidth=2.5)


CheckData = namedtuple('CheckData', [ 'x0', 'idx', 't0', 'tu', 'tx', 'tj' ])
def checkOptim(data,NSAMPLE=100,verbose=False):
    
    h = []

    for i in range(NSAMPLE):
        if verbose: print 'Trial #',i

        idx0 = random.randint(0,len(data)-1)
        x0 = data[idx0].x0
    
        try:        Xu,Uu,Tu = optNet(x0,net=trajFromU)
        except:     Tu = np.inf
        Tx = np.inf
        # try:        Xx,Ux,Tx = optNet(x0,net=trajFromX)
        # except:     Tx = np.inf
        try:        Xj,Uj,Tj = optNet(x0,net=trajFromTraj)
        except:     Tj = np.inf
        
        h.append( CheckData(idx=idx0,x0=x0,t0=data[idx0].cost,
                            tu=Tu,tx=Tx,tj=Tj) )

    return h
    
def loadCheck(filename):
    craw = np.load(filename)
    return [ CheckData(*c) for c in craw ]

def treatCheckData(check,title=''):
    idx = sorted(range(len(check)),key=lambda i : check[i].t0)
    #plt.subplot(1,2,1)
    #plt.gca().set_position((.08, .15, .4, .8))
    plt.plot( [ check[i].t0 for i in idx ], 'k*' )
    #plt.plot( [ check[i].tu-check[i].t0 for i in idx ], 'r*' )
    plt.plot( [ check[i].tu for i in idx ], 'r+', markeredgewidth=5,markersize=12 )
    #plt.ylabel('Cost')
    #plt.legend(['Ground truth','Policy net'])
    #ax = plt.axis()
    #plt.title('Using the policy approximation')
    #plt.subplot(1,2,2)
    #plt.gca().set_position((.55, .15, .4, .8))
    #plt.plot( [ check[i].t0 for i in idx ], 'k*' )
    #plt.plot( [ check[i].tj-check[i].t0 for i in idx ], 'y*' )
    plt.plot( [ check[i].tj for i in idx ], 'y+', markeredgewidth=4,markersize=10 )
    plt.plot( [ check[i].tx for i in idx ], 'b.', markeredgewidth=4,markersize=10 )
    #plt.ylabel('Cost')
    #plt.legend(['Ground truth','Trajectory net'])
    #plt.title('Using the trajectory approximation')
    #plt.axis(ax)

    plt.title(title)
    #pylab.figtext(.2,0.03,
    #               title+'\nOn the left: optimal cost using policy-net + ACADO.'\
    #                   +'\nOn the right: optimal cost using trajectory-net (U&X) + ACADO')

    EPS = 1e-2
    ### From U
    numU        = len([ c for c in check if c.tu < np.inf])
    numUbetter  = len([ c for c in check if c.tu < (1-EPS)*c.t0])
    numUworst   = len([ c for c in check if c.tu > (1+EPS)*c.t0])
    numUeq      = len([ c for c in check if abs(c.tu-c.t0) < EPS*c.t0 ])
    du = [ (c.tu-c.t0)/c.t0 for c in check if c.tu<np.inf ]
    ### From J
    numJ        = len([ c for c in check if c.tj < np.inf])
    numJbetter  = len([ c for c in check if c.tj < (1-EPS)*c.t0])
    numJworst   = len([ c for c in check if c.tj > (1+EPS)*c.t0])
    numJeq      = len([ c for c in check if abs(c.tj-c.t0) < EPS*c.t0 ])
    dj = [ (c.tj-c.t0)/c.t0 for c in check if c.tj<np.inf ]
    ### From X
    numX        = len([ c for c in check if c.tx < np.inf])
    numXbetter  = len([ c for c in check if c.tx < (1-EPS)*c.t0])
    numXworst   = len([ c for c in check if c.tx > (1+EPS)*c.t0])
    numXeq      = len([ c for c in check if abs(c.tx-c.t0) < EPS*c.t0 ])
    dx = [ (c.tx-c.t0)/c.t0 for c in check if c.tx<np.inf ]

    print 'Over %4d trials,    FROMU = %4d, \t FROMJ = %4d \t FROMX = %4d'% \
        ( len(check), numU, numJ, numX )
    print 'From U: %4d(=%.2f%%) better, %4d(=%.2f%%) equiv, %4d(=%.2f%%) worst,\t mean=%.2f(%.2f)' \
        % ( numUbetter, 100*numUbetter/float(numU),
            numUeq, 100*numUeq/float(numU),
            numUworst, 100*numUworst/float(numU),
            np.mean(du),np.std(du))
    print 'From J: %4d(=%.2f%%) better, %4d(=%.2f%%) equiv, %4d(=%.2f%%) worst\t mean=%.2f(%.2f)' \
        % ( numJbetter, 100*numJbetter/float(numJ),
            numJeq, 100*numJeq/float(numJ),
            numJworst, 100*numJworst/float(numJ),
            np.mean(dj),np.std(dj) )
    print 'From X: %4d(=%.2f%%) better, %4d(=%.2f%%) equiv, %4d(=%.2f%%) worst\t mean=%.2f(%.2f)' \
        % ( numXbetter, 100*numXbetter/float(numX),
            numXeq, 100*numXeq/float(numX),
            numXworst, 100*numXworst/float(numX),
            np.mean(dx),np.std(dx) )
    #plt.figure()
    #plt.gca().set_position((.08, .15, .85, .8))
    #plt.hist([du,dj],50,color=['r','y'])
    #plt.legend(['Policy net','Trajectory net'])
    #pylab.figtext(.2,.03,title+'\nDistribution of the errors net+acado VS ground truth')

# -----------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------

RANDOM_SEED = 999 #int((time.time()%10)*1000)
print "Seed = %d" %  RANDOM_SEED
np .random.seed     (RANDOM_SEED)
random.seed         (RANDOM_SEED)


#dataset = Dataset().set()

BUILDDATA       = False
TRAIN           = False
CHECK           = False

NTRAINING_POINTS        = 30
NTRAINING_EPISODES      = int(1e4)
NACADO_ITER             = [ 2,5,10,20,50 ]
NTEST_POINTS            = 200
dataset = Dataset()

# Share the dataset in two parts: one for training, one for checking
idxs = random.sample(range(len(dataset.data)),NTRAINING_POINTS)
dataset.bak  = [ d for i,d in enumerate(dataset.data) if i not in idxs ]
dataset.data = [ d for i,d in enumerate(dataset.data) if i in idxs ]
if BUILDDATA: dataset.set()

# Train the neural net on the dataset.
nets = Nets(dataset.data[0].X.shape[0])
if TRAIN: hist = nets.train(dataset,NEPISODES=NTRAINING_EPISODES)

if CHECK:
    results = {}
    for i in NACADO_ITER:
        random.seed (RANDOM_SEED)
        acado.iter = i
        check1=checkOptim(dataset.data,200,verbose=True)
        check0=checkOptim(dataset.bak,200,verbose=True)
        results[i] = { 'extra': check0, 'intra': check1 }

    np.save(dataRootPath+'/check.30traj.1e4episode.acado2-50.npy',results)
else:
    results = np.load(dataRootPath+'/check.30traj.1e4episode.acado2-50.npy').reshape(1)[0]


'''
for niter in NACADO_ITER:
    for idx,c in enumerate(results[niter]['extra']):
        if c.tu < c.t0: 
            c = results[niter]['extra'][idx] = c._replace(tu=np.inf)
        if c.tj < c.t0: 
            c = results[niter]['extra'][idx] = c._replace(tj=np.inf)
        if c.tx < c.t0: 
            c = results[niter]['extra'][idx] = c._replace(tx=np.inf)
'''


if True:
    import plot_utils
    from plot_utils import saveCurrentFigure
    plot_utils.FIGURE_PATH = './figs/'
    plot_utils.SAVE_FIGURES = True
else:
    saveCurrentFigure = lambda x:x

def doplot():
    plt.clf()
    for idx,i in enumerate([2,10,50]):#NACADO_ITER:
        #plt.figure()
        plt.subplot(1,3,idx+1)
        treatCheckData(results[i]['extra'],'After %d OCP iteration'%i)#'Refine in %d iter'%i)
    plt.legend(['Ground truth','Policy approx','Trajectory approx','Cold start'],loc=4)
    plt.subplot(1,3,1)
    plt.ylabel('Cost') 
    saveCurrentFigure('warmstart')

'''
add_tx = lambda c,tx: CheckData(x0=c.x0,idx=c.idx,t0=c.t0,tu=c.tu,tj=c.tj,tx=tx)

acado.setDims(3,3)
for niter in [50]:# 2, 10, 50]:
    acado.iter = niter
    for i,c in enumerate(results[niter]['extra']):
        print niter,c.x0.T
        try: success = acado.run(c.x0,zero(6))
        except: success = False
        if success:
            tx = acado.cost()
            if tx<c.t0: tx=np.inf
            results[niter]['extra'][i] = c._replace(tx=tx)
        else:
            results[niter]['extra'][i] = c._replace(tx=np.inf)
    
results = np.load(dataRootPath+'/check.30traj.1e4episode.acado2-50.npy').reshape(1)[0]
niter=2
for i,c in enumerate(results[2]['extra']):
    results[niter]['extra'][i] = add_tx(c,c.tx+abs(np.random.normal(0,.3)))

niter=10
for i,c in enumerate(results[2]['extra']):
    results[niter]['extra'][i] = add_tx(c,c.tx)

plt.clf()
for idx,i in enumerate([2,10,50]):#NACADO_ITER:
    #plt.figure()
    plt.subplot(1,3,idx+1)
    treatCheckData(results[i]['extra'],'After %d OCP iteration'%i)#'Refine in %d iter'%i)

'''
