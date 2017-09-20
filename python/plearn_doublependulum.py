from pinocchio.utils import *
from numpy.linalg import inv, norm
import math
import time
import random
import matplotlib.pylab as plt
from collections import namedtuple
import pylab
from prm import PRM,Graph,NearestNeighbor,DistanceSO3
from oprm import OptimalPRM
from specpath import acadoBinDir, acadoTxtPath
from acado_connect import AcadoConnect
from doublependulum_steering import env,acado,config,GraphDoublePendulum,ConnectAcado,DoublePendulumStateDiff,dataRootPath

plt.ion()

RANDOM_SEED = 666 #int((time.time()%10)*1000)
print "Seed = %d" %  RANDOM_SEED
np .random.seed     (RANDOM_SEED)
random.seed         (RANDOM_SEED)


# --- DATA ---------------------------------------------------------
# --- DATA ---------------------------------------------------------
# --- DATA ---------------------------------------------------------


def subsample(X, N):
    '''Subsample in N iterations the trajectory X. The output is a 
    trajectory similar to X with N points. '''
    nx = X.shape[0]
    idx = np.arange(float(N)) / (N - 1) * (nx - 1)
    hx = []
    for i in idx:
        i0 = int(np.floor(i))
        i1 = int(np.ceil(i))
        di = i % 1
        x = X[i0, :] * (1 - di) + X[i1, :] * di
        hx.append(x)
    return np.vstack(hx)


class Dataset:
    def __init__(self, graph):
        self.graph=graph

    def set(self):
        graph = self.graph

        x0s      = []        # init points
        x1s      = []        # term points
        vs       = []        # values
        us       = []        # controls
        trajxs   = []        # trajs state
        trajus   = []        # trajs state
        self.indexes = []

        import sys
        sys.stdout.write('Load dataset ')
        sys.stdout.flush()
        for (p0, p1), trajx in graph.states.items():
            sys.stdout.write('.')
            sys.stdout.flush()
            #print "Load edge", p0, p1
            traju = graph.controls[p0, p1]
            T = graph.edgeTime[p0, p1]
            DT = T / (len(trajx) - 1)
            for i0, (x0, u0) in enumerate(zip(trajx, traju)):
                for di, x1 in enumerate(trajx[i0 + 1:]):
                    if di < 5: continue
                    x0s.append(x0)
                    x1s.append(x1)
                    us.append(u0)
                    vs.append(DT * (di + 1))
                    trajxs.append(np.ravel(subsample(trajx[i0:i0 + di + 2], 20)))
                    trajus.append(np.ravel(subsample(traju[i0:i0 + di + 2], 20)))
                    self.indexes.append([p0, p1, i0, di])

        self.x0s = np.vstack(x0s)
        self.x1s = np.vstack(x1s)
        self.vs = np.vstack(vs)
        self.us = np.vstack(us)
        self.trajxs = np.vstack(trajxs)
        self.trajus = np.vstack(trajus)


# --- NETWORKS ---------------------------------------------------------
# --- NETWORKS ---------------------------------------------------------
# --- NETWORKS ---------------------------------------------------------

from networks import *


class Networks:
    BATCH_SIZE = 128

    def __init__(self,NX,NU):
        TRAJLENGTH = 20
        # bx = [10., 10., 10., 1.4, 1.4, 10., 10., 10., 2., 2.]
        # bx = bx * TRAJLENGTH
        # bx = [[-x for x in bx], bx]
        bx = np.vstack([ np.hstack([env.xmin,env.xmax]) ]*TRAJLENGTH).T
        self.bx = bx

        self.value = PolicyNetwork(NX * 2, 1).setupOptim('direct')
        self.ptrajx = PolicyNetwork(NX * 2, NX * TRAJLENGTH, umax=bx).setupOptim('direct')
        self.ptraju = PolicyNetwork(NX * 2, NU * TRAJLENGTH, umax=list(env.umax.flat)).setupOptim('direct')

        self.sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()

    def load(self, ext=''):
        tf.train.Saver().restore(self.sess, dataRootPath + '/prm' + ext)

    def save(self, ext=''):
        tf.train.Saver().save(self.sess, dataRootPath + '/prm' + ext)

    def train(self, dataset, nets=None, nepisodes=int(1e4), verbose=False, track=False):
        if nets is None: nets = [self.value, self.ptrajx, self.ptraju]

        if track:
            hist = []
            refbatch = random.sample(range(len(dataset.us)),self.BATCH_SIZE*16)
            xref     = np.hstack([dataset.x0s    [refbatch,:],
                              dataset.x1s    [refbatch,:]])
            vref     = dataset.vs     [refbatch,:]
            xsref    = dataset.trajxs [refbatch,:]
            usref    = dataset.trajus [refbatch,:]

        for episode in range(nepisodes):
            if verbose and not episode % 100:
                print 'Episode #', episode

            batch = random.sample(range(len(dataset.us)), self.BATCH_SIZE)
            xbatch = np.hstack([dataset.x0s[batch, :],
                                dataset.x1s[batch, :]])
            self.sess.run([p.optim for p in nets],
                          feed_dict={self.value.x: xbatch,
                                     self.value.uref: dataset.vs[batch, :],
                                     self.ptrajx.x: xbatch,
                                     self.ptrajx.uref: dataset.trajxs[batch, :],
                                     self.ptraju.x: xbatch,
                                     self.ptraju.uref: dataset.trajus[batch, :]})

            if track and not episode % 50:
                v = self.sess.run(self.value.policy, feed_dict={self.value.x: xref})
                xs = self.sess.run(self.ptrajx.policy, feed_dict={self.ptrajx.x: xref})
                us = self.sess.run(self.ptraju.policy, feed_dict={self.ptraju.x: xref})
                hist.append([norm(v - vref) / len(refbatch),
                             norm(us - usref) / len(refbatch),
                             norm(xs - xsref) / len(refbatch)])
                if verbose:
                    colors = ['r+', 'y*', 'b.']
                    for hi, ci in zip(hist[-1], colors):
                        plt.plot(len(hist), hi, ci)
                    plt.draw()

        if track: return hist

# --- OPTIMIZE -----------------------------------------------------
# --- OPTIMIZE -----------------------------------------------------
# --- OPTIMIZE -----------------------------------------------------


# --- ROLL OUT

def trajFromTraj(x0=None, x1=None, withPlot=None, **plotargs):
    x0 = x0.T if x0 is not None else env.sample().T
    x1 = x1.T if x1 is not None else env.sample().T
    x = np.hstack([x0, x1])
    X = nets.sess.run(nets.ptrajx.policy, feed_dict={nets.ptrajx.x: x})
    X = np.reshape(X, [max(X.shape) / env.nx, env.nx])
    U = nets.sess.run(nets.ptraju.policy, feed_dict={nets.ptraju.x: x})
    U = np.reshape(U, [max(U.shape) / env.nu, env.nu])
    T = nets.sess.run(nets.value.policy, feed_dict={nets.value.x: x})[0, 0]
    if withPlot is not None: plt.plot(X[:, 0], X[:, 1], withPlot, **plotargs)
    return X, U, T


# --- TRAJ OPT
def optNet(x0=None, x1=None, net=trajFromTraj, withPlot=False, color='r', **plotargs):
    X, U, T = net(x0, x1)
    if withPlot:
        plt.plot(X[:, 0], X[:, 1], '--', color=color, **plotargs)

    # ts = np.arange(0.,1.,1./X.shape[0])
    ts = np.arange(0., X.shape[0]) / (X.shape[0] - 1)

    np.savetxt(acado.stateFile('i'), np.vstack([ts, X.T]).T)
    np.savetxt(acado.controlFile('i'), np.vstack([ts, U.T]).T)
    acado.setTimeInterval(T)
    acado.options['Tmax'] = T * 10
    if not acado.run(x0, x1, autoInit=False):
        raise RuntimeError("Error: Acado returned false")

    Xa = acado.states()
    Ua = acado.controls()
    Ta = acado.opttime()

    if withPlot:
        plt.plot(Xa[:, 0], Xa[:, 1], '-', color=color, **plotargs)

    return Xa, Ua, Ta


# --- PLAY WITH DATA ---------------------------------------------------

def plotGrid(nets, theta=0, idxs = { 0: [-1,1], 1: [-1,1]}, x0 = None,step=1e-2):
    from grid_policy import GridPolicy
    grid = GridPolicy()
    x0 = x0 if x0 is not None else zero(env.nx)
    xm = x0.copy(); xM = x0.copy()+step/10
    for i,[vm,vM] in idxs.items(): xm[i]=vm; xM[i]= vM
    X0 = grid.setGrid(xm,xM, 1e-2)
    V = nets.sess.run(nets.value.policy, feed_dict={nets.value.x: np.hstack([X0, 0 * X0])})
    plt.scatter(X0[:, 0].flat, X0[:, 1].flat, c=V.flat, linewidths=0)
    X0 = grid.setGrid(xm,xM,step*8)
    for x in nets.sess.run(nets.ptrajx.policy, feed_dict={nets.ptrajx.x: np.hstack([X0, 0 * X0])}):
        X = np.reshape(x, [20, env.nx])
        plt.plot(X[:, 0], X[:, 1])


# def trial():
#     graph = dataset.prm.graph
#     while True:
#         i0 = random.randint(0,len(graph.x)-1)
#         if len(graph.children[i0])>0: break
#     i1 = random.sample(graph.children[i0],1)[0]
#     print 'Trial from %d to %d' % (i0,i1)
#     x0 = graph.x[i0]
#     x1 = graph.x[i1]
#     Xg = graph.states[i0,i1]
#     Xr = np.reshape(sess.run(trajx.policy,feed_dict={ trajx.x: np.hstack([x0.T,x1.T]) }),[20,6])
#     plt.plot(Xg[:,0],Xg[:,1],'r+-')
#     plt.plot(Xr[:,0],Xr[:,1],'y+-',markeredgewidth=5)
#     plt.legend('From PRM','From NNet')


def trial(i0=None, i1=None):
    graph = dataset.prm.graph
    if i0 is None:
        while True:
            i0 = random.randint(0, len(graph.x) - 1)
            if len(graph.children[i0]) > 0: break
    if i1 is None:
        i1 = random.sample(graph.children[i0], 1)[0]
    print 'Trial from %d to %d' % (i0, i1)
    x0 = graph.x[i0]
    x1 = graph.x[i1]
    Xg = graph.states[i0, i1]
    plt.plot(Xg[:, 0], Xg[:, 1], 'r*-')
    Xo, Uo, To = optNet(x0, x1, withPlot=True, color='r')


# --- IREPA ALGORTHM ---------------------------------------------------
# --- IREPA ALGORTHM ---------------------------------------------------
# --- IREPA ALGORTHM ---------------------------------------------------

def checkPrm(EPS=.05, verbose=True):
    '''Return a patch that improve the PRM edge cost.'''
    res = []
    for i0, x0 in enumerate(graph.x):
        for i1 in graph.children[i0]:
            x1 = graph.x[i1]
            Tp = graph.edgeCost[i0, i1]
            try:
                Xa, Ua, Ta = optNet(x0, x1)
            except:
                continue
            if Ta < (1 - EPS) * Tp:
                if verbose: print "Better connection: %d to %d (%.2f vs %.2f)" % (i0, i1, Ta, Tp)
                res.append([i0, i1, Xa, Ua, Ta])
    return res


def plotPrmUpdate(newTrajs, NSAMPLE=10):
    '''Plot the PRM patch computed by checkPRM'''
    for i0, i1, Xa, Ua, Ta in random.sample(newTrajs, NSAMPLE):
        x0 = graph.x[i0]
        x1 = graph.x[i1]
        Xp = graph.states[i0, i1]
        plt.plot(x0[0, 0], x0[1, 0], '*')
        plt.plot(x1[0, 0], x1[1, 0], '*')
        plt.plot(Xa[:, 0], Xa[:, 1], 'b')
        plt.plot(Xp[:, 0], Xp[:, 1], 'y', linewidth=2)


def updatePrm(newTrajs):
    '''
    Apply the PRM patch computed by checkPRM by replacing the edges with the
    new traj and cost.
    '''
    for i0, i1, Xa, Ua, Ta in newTrajs:
        graph.states[i0, i1] = Xa
        graph.controls[i0, i1] = Ua
        graph.edgeCost[i0, i1] = Ta
        graph.edgeTime[i0, i1] = Ta


def expandPRM(prm,NSAMPLE=100,verbose=False):
    graph   = prm.graph
    connect = prm.connect
    for i in range(NSAMPLE):
        i0,i1 = random.sample(range(len(graph.x)),2)
        i1, = random.sample(range(60,len(graph.x)),1)
        x0,x1 = graph.x[i0],graph.x[i1]
        if i1 in graph.children[i0]: continue
        if verbose: print '#%d: Connecting %d to %d' % (i,i0,i1)
        if connect(x0,x1): 
            graph.addEdge(i0,i1,+1,**connect.results()._asdict())
            if verbose: print '\t\t... Yes!'

# --- STEERING METHOD USING PRM -------------------------------------------------
def value(x0,x1):
    '''Distance function between 2 elements using the value neural-net approximation.'''
    return nets.sess.run( nets.value.policy, 
                          feed_dict={ nets.value.x: np.hstack([ x0.T, x1.T]) })[0,0]

def nndist(x0,x1s):
    '''
    Distance function using the Value neural net. x0 is a single point while
    x1s is a collection of points organized by rows.
    '''
    assert(len(x0)==env.nx and x1s.shape[1]==env.nx)
    n = x1s.shape[0]
    return nets.sess.run( nets.value.policy, 
                          feed_dict={ nets.value.x: np.hstack([ np.vstack([x0.T]*n), x1s]) })[:,0]

def nnnear(x0,x1s,nneighbor=1,hdistance=None,fullSort=False):
    '''Nearest neighbor using the neural-net distance function. x0 is a single point
    while x1s is a list of points. hdistance is not used.'''
    d = nndist(x0,np.hstack(x1s).T)
    if fullSort:        return np.argsort(d)[:nneighbor]
    else:               return np.argpartition(d,nneighbor)[:nneighbor]

def nnguess(x0,x1,*dummyargs):
    '''Initial guess for acado using NNet.'''
    X,U,T = trajFromTraj(x0,x1)
    N = X.shape[0]
    return X,U,np.arange(0.,N)*T/(N-1)

# --- ALGO
# --- ALGO
# --- ALGO

# --- HYPER PARAMS
INIT_PRM        = False
IREPA_ITER      = 10
IREPA_START     = 4  # Start from load


# --- SETUP ACADO
# if 'icontrol' in acado.options: del acado.options['icontrol']
# if 'istate' in acado.options: del acado.options['istate']

acado.debug(False)
acado.iter = 80
config(acado,'connect')

graph   = GraphDoublePendulum()
graph.addNode(zero(env.nx))
prm     = PRM(graph,
              sampler = env.sample,
              checker = lambda x:True,
              nearestNeighbor = NearestNeighbor(DistanceSO3([1,.1])),
              connect = ConnectAcado(acado))
prm     = OptimalPRM.makeFromPRM(prm,acado=acado,stateDiff=DoublePendulumStateDiff(env.nq))
dataset = Dataset(prm.graph)
nets    = Networks(env.nx,env.nu)

def prunePrm(prm,ntrial=-1):
    from astar import astar
    randqueue = random.sample(prm.graph.states.keys(),ntrial) if ntrial>0 else prm.graph.states.keys()
    for i0,i1 in randqueue:
        cstar = prm.pathFrom(i0,i1).cost
        cprm  = prm.graph.edgeCost[i0,i1]
        if cprm > cstar: 
            print '\t\t Remove %d-%d'%(i0,i1), '\t(%.1f vs %.1f)' % (cprm,cstar)
            prm.graph.removeEdge(i0,i1)

# --- INIT PRM ---
# --- INIT PRM ---
# --- INIT PRM ---
#acado.debug()
if INIT_PRM:
    print 'Init PRM Sample'
    prm(30,10,10,True)
    print 'Connexify'
    prm.connexifyPrm(NTRIAL=100,VERBOSE=True)
    print 'Densify'
    config(acado,'traj')
    prm.densifyPrm(1000,VERBOSE=2)
    prm.graph.save(dataRootPath+'/prm60-200-2000')
else:
    #prm.graph.load(dataRootPath+'/prm30-100-1000')
    prm.graph.load(dataRootPath+'/prm60-200-2000')

prunePrm(prm)
config(acado,'traj')
acado.iter = 80
acado.guessbak = acado.guess
acado.guess = nnguess
prm.nearestNeighborbak = prm.nearestNeighbor
prm.nearestNeighbor = nnnear


# --- IREPA LOOP
# --- IREPA LOOP
# --- IREPA LOOP
hists = {}
timings = {}
if IREPA_START>0:
    nets        .load('up%02d'%IREPA_START)
    prm.graph   .load(dataRootPath+'/up%02d'%IREPA_START)
    hists =   np.load(dataRootPath+'/hist.npy',hists)
    hists = np.reshape(hists,1)[0]

for iloop in range(IREPA_START,IREPA_ITER):
    print (('--- IREPA %d ---'%iloop)+'---'*10+'\n')*3,time.ctime()
    timings[iloop] = time.ctime()
    nets        .save('up%02d'%iloop)
    prm.graph   .save(dataRootPath+'/up%02d'%iloop)
    np          .save(dataRootPath+'/hist.npy',hists)

    dataset.set()
    hists[iloop] = nets.train(dataset,nepisodes=int(5e3),track=True,verbose=True)

    trajs = checkPrm()
    #plotPrmUpdate(trajs)
    updatePrm(trajs)

    expandPRM(prm,500,verbose=True)

try:
    iloop += 1
    nets.save('up%02d'%iloop)
    graph.save(dataRootPath+'/up%02d'%iloop)
except:    pass

# --- RELOAD
# --- RELOAD
# --- RELOAD

#acado.run(np.matrix([.5,.5,.5,0,0]+[0]*5).T,zero(10))

'''
graphs = {}
for iloop in range(IREPA_ITER+1):
    graphs[iloop] = GraphDoublePendulum()
    graphs[iloop].load(dataRootPath+'/up%02d'%iloop)

nets.load('up%02d'%IREPA_ITER)
graph=graphs[IREPA_ITER]
#hists=np.load(dataRootPath+'/hist.npy').reshape(1)[0]

dataset.graph=graph
#dataset.set()
#h = nets.train(dataset,nepisodes=int(5e3),track=True,verbose=True)


#prm.graph  = graph   = GraphDoublePendulum()
#prm(10,10,10,True)
#prm.connexifyPrm(NTRIAL=100,VERBOSE=True)
#prm.densifyPrm(100,VERBOSE=2)
'''
