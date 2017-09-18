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
from quadcopter_steering import env,acado,config,GraphQuadcopter,ConnectAcado,QuadcopterStateDiff,dataRootPath

plt.ion()

# --- ENV ----------------------------------------------------------
# --- ENV ----------------------------------------------------------
# --- ENV ----------------------------------------------------------
from quadcopter import Quadcopter

env = Quadcopter(withDisplay=False)
NX = 10
NQ = 5
NV = 5
NU = 4
UBOUND = [env.umin[0], env.umax[0]]

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

        x0s = []  # init points
        x1s = []  # term points
        vs = []  # values
        us = []  # controls
        trajxs = []  # trajs state
        trajus = []  # trajs state
        self.indexes = []

        for (p0, p1), trajx in graph.states.items():
            print "Load edge", p0, p1
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

    def __init__(self):
        TRAJLENGTH = 20
        bx = [10., 10., 10., 1.4, 1.4, 10., 10., 10., 2., 2.]
        bx = bx * TRAJLENGTH
        bx = [[-x for x in bx], bx]

        self.value = PolicyNetwork(NX * 2, 1).setupOptim('direct')
        self.ptrajx = PolicyNetwork(NX * 2, NX * TRAJLENGTH, umax=bx).setupOptim('direct')
        self.ptraju = PolicyNetwork(NX * 2, NU * TRAJLENGTH, umax=[-1, 25]).setupOptim('direct')

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
            refbatch = random.sample(range(len(dataset.us)), self.BATCH_SIZE * 16)
            xref = np.hstack([dataset.x0s[refbatch, :],
                              dataset.x1s[refbatch, :]])
            vref = dataset.vs[refbatch, :]
            xsref = dataset.trajxs[refbatch, :]
            usref = dataset.trajus[refbatch, :]

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


# --- OPTIMIZE -----------------------------------------------------
# --- OPTIMIZE -----------------------------------------------------
# --- OPTIMIZE -----------------------------------------------------


# --- ROLL OUT

def trajFromTraj(x0=None, x1=None, withPlot=None, **plotargs):
    x0 = x0.T if x0 is not None else env.sample().T
    x1 = x1.T if x1 is not None else env.sample().T
    x = np.hstack([x0, x1])
    X = nets.sess.run(nets.ptrajx.policy, feed_dict={nets.ptrajx.x: x})
    X = np.reshape(X, [max(X.shape) / NX, NX])
    U = nets.sess.run(nets.ptraju.policy, feed_dict={nets.ptraju.x: x})
    U = np.reshape(U, [max(U.shape) / NU, NU])
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

def plotGrid(theta):
    from grid_policy import GridPolicy
    grid = GridPolicy()
    X0 = grid.setGrid([-1., -1., theta, 0, 0, 0], [+1, +1, theta + 1e-3, 1e-3, 1e-3, 1e-3], 1e-2)
    V = sess.run(value.policy, feed_dict={value.x: np.hstack([X0, 0 * X0])})
    plt.scatter(X0[:, 0].flat, X0[:, 1].flat, c=V.flat, linewidths=0, vmin=0, vmax=2)
    X0 = grid.setGrid([-1., -1., theta, 0, 0, 0], [+1, +1, theta + 1e-3, 1e-3, 1e-3, 1e-3], 8e-2)
    for x in sess.run(trajx.policy, feed_dict={trajx.x: np.hstack([X0, 0 * X0])}):
        X = np.reshape(x, [20, 6])
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



# --- ALGO
# --- ALGO
# --- ALGO

# --- HYPER PARAMS
INIT_PRM        = True
IREPA_ITER      = 0

# --- SETUP ACADO
# if 'icontrol' in acado.options: del acado.options['icontrol']
acado.debug(False)
acado.iter = 80
config(acado,'connect')

graph   = GraphQuadcopter()
prm     = PRM(graph,
              sampler = env.reset,
              checker = lambda x:True,
              nearestNeighbor = NearestNeighbor(DistanceSO3([1,.1])),
              connect = ConnectAcado(acado))
prm     = OptimalPRM.makeFromPRM(prm,acado=acado,stateDiff=QuadcopterStateDiff())
dataset = Dataset(prm.graph)
nets    = Networks()

updated = {(i0, i1): False for (i0, i1) in graph.states.keys()}

# --- INIT PRM ---
if INIT_PRM:
    print 'Init PRM Sample'
    prm(30,10,10,True)
    print 'Connexify'
    prm.connexifyPrm(NTRIAL=100,VERBOSE=True)
    print 'Densify'
    config(acado,'traj')
    prm.densifyPrm(100,VERBOSE=2)
    prm.graph.save(dataRootPath+'/prm30-100-100')
else:
    prm.graph.save(dataRootPath+'/prm30-100-100')

# --- IREPA LOOP
for iloop in range(IREPA_ITER):
    nets.save('up%02d'%iloop)
    prm.graph.save(dataRootPath+'/up%02d'%iloop)

    trajs = checkPrm()
    #plotPrmUpdate(trajs)
    updatePrm(trajs)

    dataset.set()
    nets.train(dataset,nepisodes=int(5e3),track=True,verbose=True)

iloop += 1
nets.save('up%02d'%iloop)
graph.save(dataRootPath+'/up%02d'%iloop)

