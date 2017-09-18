from pinocchio.utils import *
from numpy.linalg import inv,norm
import math
import time
import random
import matplotlib.pylab as plt
import matplotlib
from collections import namedtuple
import pylab
from pylab import figtext
#plt.ion()

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
from prm import PRM,Graph,NearestNeighbor,DistanceSO3
from oprm import OptimalPRM

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
    def __init__(self,graph):
        #self.prm=PRM(Graph(),*[None]*4)
        #self.prm.graph.load(dataRootPath)
        self.graph = graph

    def set(self):
        graph = self.graph

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

        if track: return hist

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


def trial(i0=None,i1=None):
    graph = dataset.graph
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

# --- UPDATE PRM -------------------------------------------------------

def checkPrm(EPS=.05,verbose=True):
    '''Return a patch that improve the PRM edge cost.'''
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
    '''Plot the PRM patch computed by checkPRM'''
    for i0,i1,Xa,Ua,Ta in random.sample(newTrajs,NSAMPLE):
        x0 = graph.x[i0]
        x1 = graph.x[i1]
        Xp = graph.states[i0,i1]
        plt.plot(x0[0,0],x0[1,0],'*')
        plt.plot(x1[0,0],x1[1,0],'*')
        plt.plot(Xa[:,0],Xa[:,1],'b')
        plt.plot(Xp[:,0],Xp[:,1],'y',linewidth=2)


def updatePrm(newTrajs):
    '''
    Apply the PRM patch computed by checkPRM by replacing the edges with the
    new traj and cost.
    '''
    for i0,i1,Xa,Ua,Ta in newTrajs:
        graph.states[i0,i1] = Xa
        graph.controls[i0,i1] = Ua
        graph.edgeCost[i0,i1] = Ta
        graph.edgeTime[i0,i1] = Ta

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

# --- MAIN ----------------------------------------------------------------------
from bicopter_steering import env,acado,config,GraphBicopter,ConnectAcado,BicopterStateDiff,dataRootPath

RANDOM_SEED = 999 #int((time.time()%10)*1000)
print "Seed = %d" %  RANDOM_SEED
np .random.seed     (RANDOM_SEED)
random.seed         (RANDOM_SEED)

config(acado,'connect',env)
graph   = GraphBicopter()
prm     = PRM(graph,
            sampler = env.reset,
            checker = lambda x:True,
            nearestNeighbor = NearestNeighbor(DistanceSO3([1,.1])),
            connect = ConnectAcado(acado))
prm = OptimalPRM.makeFromPRM(prm,acado=prm.connect.acado,stateDiff=BicopterStateDiff())

# --- INIT PRM ---
'''print 'Init PRM Sample'
prm(30,10,10,True)
print 'Connexify'
prm.connexifyPrm(NTRIAL=100,VERBOSE=True)
print 'Densify'
config(acado,'traj')
prm.densifyPrm(100,VERBOSE=2)
prm.graph.save(dataRootPath+'/prm30-100-100')
'''
prm.graph.load(dataRootPath+'/prm30-100-100')


dataset = Dataset(prm.graph)
nets    = Networks()
#nets.train(dataset,nepisodes=2000,track=True,verbose=True)
#nets.save()

def expandPRM(prm,NSAMPLE=100,verbose=False):
    graph   = prm.graph
    connect = prm.connect
    for i in range(NSAMPLE):
        i0,i1 = random.sample(range(len(graph.x)),2)
        x0,x1 = graph.x[i0],graph.x[i1]
        if i1 in graph.children[i0]: continue
        if verbose: print '#%d: Connecting %d to %d' % (i,i0,i1)
        if connect(x0,x1): 
            graph.addEdge(i0,i1,+1,**connect.results()._asdict())
            if verbose: print '\t\t... Yes!'

config(acado,'traj')
acado.iter = 20
acado.guess = nnguess

'''
hists    = {}
for iloop in range(10):

    nets.save('up%02d'%iloop)
    graph.save(dataRootPath+'/up%02d'%iloop)

    dataset.set()
    hists[iloop] = nets.train(dataset,nepisodes=int(3e3),track=True,verbose=True)

    trajs = checkPrm()
    updatePrm(trajs)

    expandPRM(prm,500,verbose=True)

iloop += 1
nets.save('up%02d'%iloop)
graph.save(dataRootPath+'/up%02d'%iloop)
'''

# --- PLOTS -----------------------------------------------------------------

if True:
    import plot_utils
    from plot_utils import saveCurrentFigure
    plot_utils.FIGURE_PATH = './figs/'
    plot_utils.SAVE_FIGURES = True
else:
    saveCurrentFigure = lambda x:x

#plt.ion()

NG=10
graphs = {}
for iloop in range(NG+1):
    graphs[iloop] = Graph()
    graphs[iloop].load(dataRootPath+'/up%02d'%iloop)
hists=np.load(dataRootPath+'/hist.npy').reshape(1)[0]

## Number of connections wrt nb steering calls in prm.
prmcounts = np.load(dataRootPath+'/prmdiffusion.npy') 
## Number of connections wrt nb steering calls in irepa.
irepacounts = np.load(dataRootPath+'/irepadiffusion.npy') 

# ---
# ---
# ---

Ts = np.matrix([ [ graphs[i].edgeTime[k] for i in range(NG) ] for k in graphs[0].edgeTime.keys() ]).T
'''
plt.subplot(2,1,1)
plt.plot(np.mean(Ts,1),'y+-',markeredgewidth=15)
plt.ylabel('Mean PRM cost')
plt.subplot(2,1,2)
hist = reduce(lambda a,b:a+b,hists.values())
plt.plot(np.arange(len(hist))/float(len(hists[0])),np.array(hist)*[100,1,5])
plt.ylabel('RMS')
plt.xlabel('IREPA iterations')
plt.legend(['V*100','Pi','X*5'])
plt.axis([0,9,0,2])
saveCurrentFigure('irepa_progress')
'''
# ---
'''
connexs = np.array([len(graphs[i].edgeTime) for i  in range(len(graphs))])
plt.plot(connexs,'b+-')
plt.axis([0,9,0,700])
plt.xlabel('IREPA iterations')
plt.ylabel('# of connections')
saveCurrentFigure('irepa_connect')
'''
# ---
# ---
# ---
'''
plt.plot(np.arange(len(irepacounts))+100,irepacounts)
plt.plot(np.arange(len(prmcounts))+100,prmcounts)
plt.xlabel('# of steering tentative')
plt.ylabel('# of connections')
plt.legend(['IREPA','PRM'])
saveCurrentFigure('prmvsirepa')
'''
# ---
# ---
# ---

def plotEvolution(i0 = None,i1=None,subrange=1,subn=0):
    if i0 is None or i1 is None:
        i0,i1 = random.sample([ k for k,v in upd.items() if v],1)[0]
        print 'Check %d to %d' % (i0,i1)
    x0,x1 = graphs[0].x[i0],graphs[0].x[i1]
    for iloop in range(len(graphs)):
        nets.load('up%02d'%iloop)
        #optNet(x0,x1,withPlot=True)
        Xn,_,Tn = trajFromTraj(x0,x1)
        try: Xa,_,Ta = optNet(x0,x1)
        except: Xa = np.zeros([0,2]); Ta=np.inf
        Xp     = graphs[iloop].states[i0,i1]
        Tp     = graphs[iloop].edgeTime[i0,i1]
        print '%d: net=%.2f, ac=%.2f prm=%.2f'% (iloop,Tn,Ta,Tp)

        c = 1-(iloop/(float(len(graphs))) + .05)
        plotargs  = { 'c':  '%.2f' % c,
                      'linewidth': c*4+.5 }
        plt.subplot(subrange,3,1+3*subn)
        plt.plot(Xp[:,0],Xp[:,1], **plotargs ) 
        if iloop==0: ax=plt.axis()
        plt.axis(ax)
        plt.title('PRM')
        plt.subplot(subrange,3,2+3*subn)
        plt.plot(Xn[:,0],Xn[:,1], **plotargs  )
        plt.title('Net')
        plt.axis(ax)
        plt.subplot(subrange,3,3+3*subn)
        plt.plot(Xa[:,0],Xa[:,1], **plotargs ) 
        plt.title('Acado')
        plt.axis(ax)
        plt.draw()

'''
#pairs   = random.sample(graphs[0].edgeTime.keys(),10)
dec     = {k:graphs[0].edgeCost[k]-graphs[NG].edgeCost[k] for k in graphs[0].edgeTime.keys() }
pairs   = map(lambda kv:kv[0],sorted(dec.items(),key=lambda kv:kv[1])[-20:])

pairs = [ [15,12], [1,24], [1,11], [11,29], [11,1],[6,18] ]
#for i in range(2): plt.figure(i); plotEvolution(*pairs[i]); figtext(.2,.2,'%d-%d'%(pairs[i]))

#for p in pairs: plotEvolution(*p)
plotEvolution(*pairs[0],subrange=3,subn=0)
plotEvolution(*pairs[4],subrange=3,subn=1)
plotEvolution(*pairs[5],subrange=3,subn=2)

for i in range(4,9): plt.subplot(3,3,i); plt.title('')
for i in range(4,7): plt.subplot(3,3,i); plt.axis([-.65,.35,-.1,1.3])
for i in range(1,4): plt.subplot(3,3,i); plt.axis([-1.05,-0.1,0.01,1.99])
saveCurrentFigure('irepa_trajectories')
'''
# ---
# ---
# ---

'''
ks = [(7, 3),  (15, 20),  (14, 4),  (17, 20),  (1, 6),  (17, 25),  (1, 11),  (6, 10),  (11, 22),  (15, 11),  (6, 25),  (6, 28),  (9, 14),  (9, 3),  (5, 24),  (11, 0),  (15, 29),  (14, 2),  (27, 22),  (17, 18),  (1, 12),  (8, 12),  (2, 11),  (4, 26),  (6, 2),  (6, 13),  (11, 15),  (7, 8),  (6, 16),  (1, 21),  (15, 13),  (22, 23),  (1, 26),  (1, 15),  (27, 11),  (9, 4),  (12, 2),  (11, 10),  (7, 24),  (14, 19),  (22, 7),  (18, 1),  (1, 10),  (8, 6),  (9, 7),  (14, 28),  (4, 28),  (11, 4),  (14, 21),  (7, 1),  (6, 11),  (11, 9),  (1, 19),  (18, 21),  (23, 24),  (22, 0),  (27, 0),  (4, 6),  (5, 7),  (11, 3),  (14, 8),  (7, 4),  (11, 20),  (15, 9),  (14, 3),  (11, 25),  (1, 3),  (8, 13),  (22, 13),  (8, 0),  (8, 19),  (14, 26),  (11, 14),  (5, 23),  (1, 20),  (1, 25),  (22, 27),  (1, 14),  (14, 1),  (5, 0),  (8, 16),  (11, 13),  (16, 11),  (14, 10),  (22, 17),  (1, 9),  (22, 11),  (4, 2),  (27, 9),  (29, 6),  (9, 27),  (15, 24),  (6, 8),  (11, 8),  (11, 29),  (17, 21),  (17, 26),  (22, 1),  (27, 3),  (9, 12),  (8, 4),  (5, 9),  (29, 28),  (6, 21),  (11, 23),  (11, 24),  (1, 29),  (1, 2),  (8, 14),  (8, 1),  (8, 20),  (11, 1),  (5, 22),  (12, 10),  (11, 18),  (1, 24),  (22, 24),  (27, 24),  (1, 13),  (8, 11),  (4, 14),  (14, 20),  (11, 12),  (16, 12),  (14, 11),  (27, 18),  (1, 27),  (22, 5),  (27, 7),  (27, 8),  (8, 27)] 
ks = [ k for k in ks if k in graphs[0].edgeTime ]


costs = np.zeros([len(graphs),len(ks)])
for ig,graph in  graphs.items():
    for ik,k in enumerate(ks):
        costs[ig,ik] = graph.edgeTime[k]

prmcosts = np.load(dataRootPath+'/prmmeancost.npy')

plt.plot(np.arange(len(graphs))*140,np.mean(costs,1))
plt.plot(np.arange(10)*300,np.mean(prmcosts,1))
'''

# ---
# ---
# ---
nets.load('up%02d'%NG)

from grid_policy import GridPolicy
grid=GridPolicy()

'''grid.setGrid([-1.,-1.,0,0,0,0],[+1,+1,1e-3,1e-3,1e-3,1e-3],25e-2)
x1 = zero(6)
Tmax = 0.
trajs = []
for x in grid.grid:
    x0 = np.matrix(x).T
    Xp,Up,Tp = trajFromTraj(x0,x1)
    Xa,Ua,Ta = optNet(x0,x1)
    trajs.append( [[x0,x1],[Xp,Up,Tp],[Xa,Ua,Ta]] )

Tm = min([ t[2][2] for t in trajs]+[ t[1][2] for t in trajs])
TM = max([ t[2][2] for t in trajs]+[ t[1][2] for t in trajs])

# for [x0,x1],[Xp,Up,Tp],[Xa,Ua,Ta] in trajs:
#     plt.figure(1)
#     print Tp,(1-Tp/T), Ta,(1-Ta/T)
#     assert((1-Tp/T)>=0 and (1-Tp/T)<=1)
#     plt.plot(Xp[:,0],Xp[:,1],c='%.2f'%(1-Tp/T))
#     plt.figure(2)
#     assert((1-Tp/T)>=0 and (1-Ta/T)<=1)
#     plt.plot(Xa[:,0],Xa[:,1],c='%.2f'%(1-Ta/T))
#     plt.draw()


norm = matplotlib.colors.Normalize(vmin=Tm,    vmax=TM)
c_m = matplotlib.cm.rainbow
s_m = matplotlib.cm.ScalarMappable(cmap=c_m, norm=norm)
s_m.set_array([])

for [x0,x1],[Xp,Up,Tp],[Xa,Ua,Ta] in trajs:
    plt.subplot(1,2,1)
    plt.plot(Xp[:,0],Xp[:,1],color=s_m.to_rgba(Tp))
    plt.subplot(1,2,2)
    plt.plot(Xa[:,0],Xa[:,1],color=s_m.to_rgba(Ta))

plt.colorbar(s_m)
plt.subplot(1,2,1)
plt.title('Approximate trajectories')
plt.subplot(1,2,2)
plt.title('Refined trajectories')
    
saveCurrentFigure('bundle')
'''

# ---
# ---
# ---
'''
grid.setGrid([-1.,-1.,0,0,0,0],[+1,+1,1e-3,1e-3,1e-3,1e-3],2e-2)
x0s = grid.grid
x1  = np.zeros(6)
x1s = np.vstack( [x1]*x0s.shape[0] )
xs  = np.hstack([x0s,x1s])

#fig, axes = plt.subplots(nrows=2, ncols=2)
fig = plt.figure()

for iplot,inet in enumerate([1,4,7,9]):
    plt.subplot(2,2,iplot+1)
    plt.xlabel('Iteration #%d'%inet)
    nets.load('up%02d'%inet)
    vs  = nets.sess.run(nets.value.policy,feed_dict={nets.value.x:xs})
    us  = nets.sess.run(nets.ptraju.policy,feed_dict={nets.ptraju.x:xs})[:,:2]
    #im = plt.scatter(xs[:,0].flat,xs[:,1].flat,c=vs.flat,linewidths=0,s=100,vmin=0,vmax=2)
    im = plt.scatter(xs[:,0].flat,xs[:,1].flat,c=(us[:,0]+1).flat,linewidths=0,s=100,vmin=0,vmax=25)

fig.subplots_adjust(right=0.85)
cbar_ax = fig.add_axes([0.9, 0.15, 0.02, 0.7])
fig.colorbar(im, cax=cbar_ax)
#saveCurrentFigure('scatter_value')
saveCurrentFigure('scatter_policy')
'''
# ---
# ---
# ---

'''

dec = sorted({ k: graphs[6].edgeCost[k]-graphs[0].edgeCost[k] for k in graphs[0].states.keys() }.items(),
             key=lambda x,y:y)
for (i0,i1),d in dec[:10]:
    plt.clf()
    plotEvolution(i0,i1)
    raw_input('See next ?')

'''
# ---
# ---
# ---
'''
env.initDisplay()

acado.iter=200
acado.options['steps']=30

x0=np.matrix([2,1,0,0,0,0.]).T

X,U,T = optNet(x0,zero(6))

p = 25

x1 = X[p:p+1,:].T
X = X[:p,:]
U = U[:p,:]
T = T*float(acado.options['steps'])/p

v  = np.matrix([ 0,0,0, 1.4, -1.3, .2 ]).T
x1 += v
X1,U1,T1 = optNet(x1,zero(6))

N1 = int(T1/T*p)
X1 = subsample(X1,N1)
U1 = subsample(U1,N1)

X12 = np.vstack([X,X1])
X12 = subsample(X12,X12.shape[0]*5)

gui = env.viewer.viewer.gui
try:gui.addSphere('world/ball',.05,[1.,0.0,0,1])
except:pass

cam = [0.29577651619911194, -4.202201843261719, 0.8779348731040955, 0.7389997243881226, 0.6707877516746521, -0.03221357613801956, -0.05371693894267082]
gui.setCameraTransform(0,cam)
env.display(X12[0,:])

#ballstart = np.matrix([ 1.,1.,1. ]).T
tstart    = 80
def xz2xyz(xz):
    res = zero(3)
    res[[0,2]] = xz[:2]
    return res

ballend   = xz2xyz(x1)
ballstart    = ballend - xz2xyz(v[3:])
ballstart[2] += .5
tend      = p*5

gui.startCapture(0,'./figs/captures/bicopterball','png')
gui.setVisibility('world/ball','OFF')
for t,x in enumerate(X12):
    if t==tstart:        gui.setVisibility('world/ball','ON')
    if t>tend+1:          gui.setVisibility('world/ball','OFF')
    if t>=tstart and t<=tend:
        ball = ballstart + (ballend-ballstart)*float(t-tstart)/(tend-tstart)
        gui.applyConfiguration('world/ball', ball.A1.tolist()+[1.,0.,0.,0.])
    env.display(x.T)
    time.sleep(.05)
gui.stopCapture(0)
'''
