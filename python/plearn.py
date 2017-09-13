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

dataRootPath = 'data/planner/bicopter'
from prm import PRM,Graph


def subsample(X,N):
    '''Subsample in N iterations the trajectory X. The output is a 
    trajectory similar to X with N points. '''
    nx  = X.shape[0]
    idx = np.arange(float(N))/(N-1)*(nx-1)
    hx  = []
    for i in idx:
        i0 = int(np.floor(i))
        i1 = int(np.ceil(i))
        di = i%1
        x  = X[i0,:]*(1-di) + X[i1,:]*di
        hx.append(x)
    return np.vstack(hx)



class Dataset:
    def __init__(self,datafile=dataRootPath):
        self.prm=PRM(Graph(),*[None]*4)
        self.prm.graph.load(dataRootPath)

    def set(self):
        graph = self.prm.graph

        x0s      = []        # init points
        x1s      = []        # term points
        vs       = []        # values
        us       = []        # controls
        trajxs   = []        # trajs state
        trajus   = []        # trajs state
        self.indexes = []

        for (p0,p1),trajx in graph.states.items():
            print "Load edge",p0,p1
            traju = graph.controls[p0,p1]
            T     = graph.edgeTime[p0,p1]
            DT    = T/(len(trajx)-1)
            for i0,(x0,u0) in enumerate(zip(trajx,traju)):
                for di,x1 in enumerate(trajx[i0+1:]):
                    if di<5: continue
                    x0s.append(x0)
                    x1s.append(x1)
                    us .append(u0)
                    vs .append(DT*(di+1))
                    trajxs.append( np.ravel(subsample(trajx[i0:i0+di+2],20)) )
                    trajus.append( np.ravel(subsample(traju[i0:i0+di+2],20)) )
                    self.indexes.append( [p0,p1,i0,di] )

        self.x0s        = np.vstack(x0s)
        self.x1s        = np.vstack(x1s)
        self.vs         = np.vstack(vs )
        self.us         = np.vstack(us )
        self.trajxs     = np.vstack(trajxs)
        self.trajus     = np.vstack(trajus)


# --- NETWORKS ---------------------------------------------------------
# --- NETWORKS ---------------------------------------------------------
# --- NETWORKS ---------------------------------------------------------

from networks import *

class Networks:
    BATCH_SIZE = 128

    def __init__(self):
        TRAJLENGTH = 20
        bx     = [ 5.,5.,np.pi/2, 10.,10.,10. ]
        bx     = bx*TRAJLENGTH
        bx     = [ [ -x for x in bx ], bx ]
 
        self.value  = PolicyNetwork(NX*2,1)                           .setupOptim('direct')
        self.ptrajx = PolicyNetwork(NX*2,NX*TRAJLENGTH,umax=bx)       .setupOptim('direct')
        self.ptraju = PolicyNetwork(NX*2,NU*TRAJLENGTH,umax=[-1,25])  .setupOptim('direct')

        self.sess   = tf.InteractiveSession()
        tf.global_variables_initializer().run()
    
    def load(self,ext=''):
        tf.train.Saver().restore(self.sess,dataRootPath+'/prm'+ext)
    def save(self,ext=''):
        tf.train.Saver().save   (self.sess,dataRootPath+'/prm'+ext)

    def train(self,dataset,nets=None,nepisodes = int(1e4),verbose = False, track = False):
        if nets is None: nets = [ self.value,self.ptrajx,self.ptraju ]

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
                print 'Episode #',episode

            batch  = random.sample(range(len(dataset.us)),self.BATCH_SIZE)
            xbatch = np.hstack([ dataset.x0s    [batch,:],
                                 dataset.x1s    [batch,:]])
            self.sess.run([ p.optim for p in nets],
                          feed_dict={ self.value.x        : xbatch,
                                      self.value .uref    : dataset.vs    [batch,:],
                                      self.ptrajx.x       : xbatch,
                                      self.ptrajx.uref    : dataset.trajxs [batch,:],
                                      self.ptraju.x       : xbatch,
                                      self.ptraju.uref    : dataset.trajus [batch,:] })
            
            if track and not episode % 50:
                v  = self.sess.run( self.value .policy, feed_dict={ self.value .x: xref })
                xs = self.sess.run( self.ptrajx.policy, feed_dict={ self.ptrajx.x: xref })
                us = self.sess.run( self.ptraju.policy, feed_dict={ self.ptraju.x: xref })
                hist.append( [ norm(v -vref )/len(refbatch), 
                               norm(us-usref)/len(refbatch), 
                               norm(xs-xsref)/len(refbatch)] )
                if verbose: 
                    colors = [ 'r+', 'y*', 'b.' ]
                    for hi,ci in zip(hist[-1],colors): 
                        plt.plot(len(hist),hi,ci)
                    plt.draw()


# --- OPTIMIZE -----------------------------------------------------
# --- OPTIMIZE -----------------------------------------------------
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

# --- ROLL OUT

def trajFromTraj(x0 = None,x1 = None, withPlot=None,**plotargs):
    x0 = x0.T if x0 is not None else env.sample().T
    x1 = x1.T if x1 is not None else env.sample().T
    x = np.hstack([x0,x1])
    X = nets.sess.run(nets.ptrajx.policy,feed_dict={ nets.ptrajx.x: x })
    X = np.reshape(X,[max(X.shape)/NX,NX])
    U = nets.sess.run(nets.ptraju.policy,feed_dict={ nets.ptraju.x: x })
    U = np.reshape(U,[max(U.shape)/NU,NU])
    T = nets.sess.run(nets.value.policy,feed_dict={ nets.value.x:   x })[0,0]
    if withPlot is not None: plt.plot(X[:,0],X[:,1],withPlot,**plotargs)
    return X,U,T

# --- TRAJ OPT
def optNet(x0=None,x1=None,net=trajFromTraj,withPlot=False,color='r',**plotargs):
    X,U,T = net(x0,x1)
    if withPlot:
        plt.plot(X [:,0],X [:,1],'--',color=color,**plotargs)

    #ts = np.arange(0.,1.,1./X.shape[0])
    ts = np.arange(0.,X.shape[0])/(X.shape[0]-1)

    np.savetxt(acado.stateFile  ('i'),  np.vstack([ ts, X.T]).T )
    np.savetxt(acado.controlFile('i'),  np.vstack([ ts, U.T]).T )
    acado.setTimeInterval(T)
    acado.options['Tmax'] = T*10
    if not acado.run(x0,x1,autoInit=False):
        raise RuntimeError("Error: Acado returned false")

    Xa = acado.states()
    Ua = acado.controls()
    Ta = acado.opttime()

    if withPlot:
        plt.plot(Xa[:,0],Xa[:,1],'-', color=color,**plotargs)
        
    return Xa,Ua,Ta


# --- PLAY WITH DATA ---------------------------------------------------

def plotGrid(theta):
    from grid_policy import GridPolicy
    grid = GridPolicy()
    X0 = grid.setGrid([-1.,-1.,theta,0,0,0],[+1,+1,theta+1e-3,1e-3,1e-3,1e-3],1e-2)
    V  = sess.run(value.policy,feed_dict={ value.x: np.hstack([X0,0*X0]) })
    plt.scatter(X0[:,0].flat,X0[:,1].flat,c=V.flat,linewidths=0,vmin=0,vmax=2)
    X0 = grid.setGrid([-1.,-1.,theta,0,0,0],[+1,+1,theta+1e-3,1e-3,1e-3,1e-3],8e-2)
    for x in sess.run(trajx.policy,feed_dict={ trajx.x: np.hstack([X0,0*X0]) }):
        X = np.reshape(x,[20,6])
        plt.plot(X[:,0],X[:,1])


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


def trial(i0=None,i1=None):
    graph = dataset.prm.graph
    if i0 is None:
        while True:
            i0 = random.randint(0,len(graph.x)-1)
            if len(graph.children[i0])>0: break
    if i1 is None:
        i1 = random.sample(graph.children[i0],1)[0]
    print 'Trial from %d to %d' % (i0,i1)
    x0 = graph.x[i0]
    x1 = graph.x[i1]
    Xg = graph.states[i0,i1]
    plt.plot(Xg[:,0],Xg[:,1],'r*-')
    Xo,Uo,To=optNet(x0,x1,withPlot=True,color='r')


def checkPrm(EPS=.05,verbose=True):
    res = []
    for i0,x0 in enumerate(graph.x):
        for i1 in graph.children[i0]:
            x1 = graph.x[i1]
            Tp = graph.edgeCost[i0,i1]
            try:            Xa,Ua,Ta = optNet(x0,x1)
            except:         continue
            if Ta < (1-EPS)*Tp:
                if verbose: print "Better connection: %d to %d (%.2f vs %.2f)" % (i0,i1,Ta,Tp)
                res.append( [i0,i1, Xa,Ua,Ta] )
    return res

def plotPrmUpdate(newTrajs,NSAMPLE=10):
    for i0,i1,Xa,Ua,Ta in random.sample(newTrajs,NSAMPLE):
        x0 = graph.x[i0]
        x1 = graph.x[i1]
        Xp = graph.states[i0,i1]
        plt.plot(x0[0,0],x0[1,0],'*')
        plt.plot(x1[0,0],x1[1,0],'*')
        plt.plot(Xa[:,0],Xa[:,1],'b')
        plt.plot(Xp[:,0],Xp[:,1],'y',linewidth=2)


def updatePrm(newTrajs):
    for i0,i1,Xa,Ua,Ta in newTrajs:
        graph.states[i0,i1] = Xa
        graph.controls[i0,i1] = Ua
        graph.edgeCost[i0,i1] = Ta
        graph.edgeTime[i0,i1] = Ta


dataset = Dataset()
dataset.set()

nets = Networks()
plt.figure(1)
nets.train(dataset,track=True,verbose=True)
nets.save()

graph=dataset.prm.graph
acado.iter = 20

updated = { (i0,i1): False for (i0,i1) in graph.states.keys() }

'''
for iloop in range(6):
    nets.save('up%02d'%iloop)
    graph.save(dataRootPath+'/up%02d'%iloop)

    trajs = checkPrm()
    for i0,i1,_,_,_ in trajs:        updated[i0,i1] = True

    #plt.figure(2)
    #plotPrmUpdate(trajs)
    updatePrm(trajs)

    #plt.figure(1)
    dataset.set()
    nets.train(dataset,nepisodes=int(5e3),track=True,verbose=True)
    
iloop += 1
nets.save('up%02d'%iloop)
graph.save(dataRootPath+'/up%02d'%iloop)
'''

graphs = {}
for iloop in range(7):
    graphs[iloop] = Graph()
    graphs[iloop].load(dataRootPath+'/up%02d'%iloop)
Ts = np.matrix([ [ graphs[i].edgeTime[k] for i in graphs.keys() ] for k in graphs[0].edgeTime.keys() if updated[k] ]).T



def plotEvolution(i0 = None,i1=None):
    if i0 is None or i1 is None:
        i0,i1 = random.sample([ k for k,v in upd.items() if v],1)[0]
        print 'Check %d to %d' % (i0,i1)
    x0,x1 = graph.x[i0],graph.x[i1]
    for iloop in range(7):
        nets.load('up%02d'%iloop)
        #optNet(x0,x1,withPlot=True)
        Xn,_,Tn = trajFromTraj(x0,x1)
        try: Xa,_,Ta = optNet(x0,x1)
        except: Xa = np.zeros([0,2]); Ta=np.inf
        Xp     = graphs[iloop].states[i0,i1]
        Tp     = graphs[iloop].edgeTime[i0,i1]
        print '%d: net=%.2f, ac=%.2f prm=%.2f'% (iloop,Tn,Ta,Tp)

        c = 1-(iloop/7.+.05)
        plotargs  = { 'c':  '%.2f' % c,
                      'linewidth': c*4+.5 }
        plt.subplot(2,2,0)
        plt.plot(Xn[:,0],Xn[:,1], **plotargs  )
        plt.title('Net')
        plt.subplot(2,2,1)
        plt.plot(Xa[:,0],Xa[:,1], **plotargs ) 
        plt.title('Acado')
        plt.subplot(2,2,2)
        plt.plot(Xp[:,0],Xp[:,1], **plotargs ) 
        plt.title('PRM')
        plt.draw()


dec = sorted({ k: graphs[6].edgeCost[k]-graphs[0].edgeCost[k] for k in graphs[0].states.keys() }.items(),
             key=lambda x,y:y)
for (i0,i1),d in dec[:10]:
    plt.clf()
    plotEvolution(i0,i1)
    raw_input('See next ?')

