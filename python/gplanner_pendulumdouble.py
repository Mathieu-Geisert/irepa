from pinocchio.utils import *
from numpy.linalg import inv,norm
import math
import time
import random
import matplotlib.pylab as plt

from prm import *
from oprm import OptimalPRM
from grid_policy import GridPolicy
from cursor_tricks import FromCursor

from specpath import acadoPath,dataRootPath,acadoTxtPath
from pendulum import Pendulum
from acado_connect import AcadoConnect
     
# --- PENDULUM DOUBLE ------------------------------------------------------------
# --- PENDULUM DOUBLE ------------------------------------------------------------
# --- PENDULUM DOUBLE ------------------------------------------------------------

class GraphPendulumDouble(Graph):
     '''Specialization of Graph to have the display dedicated to double pendulum.'''
     def __init__(self): Graph.__init__(self)

     def plotState(self,q,v=None,color='r'):
          if v is None: v= q[2:]; q = q[:2]
          SCALE = 10.
          l = norm(v)
          if l<0.05: return
          plt.arrow(q[0,0],q[1,0],v[0,0]/SCALE,v[1,0]/SCALE,
                    fc='r', ec='r', alpha=0.5, width=0.02, head_width=0.05, head_length=0.05, 
                    head_starts_at_zero=False, shape='right',length_includes_head=True)
     def plotNode(self,idx,plotstr='k+',**kwargs):
          x = self.x[idx]
          if x is None: return
          self.plotState(x[:2],x[2:],color=plotstr)
          plt.plot(x[0,0],x[1,0],plotstr,**kwargs)

     def plotEdge(self,idx1,idx2,plotstr='k',withTruePath = False,**kwargs):
          path = self.states[idx1,idx2]
          if withTruePath: 
               plt.plot(path[:,0],path[:,1],plotstr,**kwargs)
          else: 
               plt.plot(path[[0,-1],0],path[[0,-1],1],plotstr,**kwargs)

def config(acado,label,env=None):
     if env is not None:
          acado.options['friction'] = \
              "{0:f} {1:f}".format([env.Kf,]*2) if isinstance(env.Kf,float) \
              else  "{0:f} {1:f}".format(*env.Kf.diagonal())
          acado.options['umax']     = "%.2f %.2f" % tuple([x for x in env.umax])
          acado.options['armature'] = env.armature
          
     if label == "connect":
          acado.debug(False)
          acado.iter                = 20
          acado.options['steps']    = 25
          acado.setTimeInterval(1.5)

     elif label == "traj":
          del acado.options['horizon']
          del acado.options['Tmax']
          acado.debug(False)
          acado.iter                = 100
          acado.options['steps']    = 50
          
     elif label == "policy":
          del acado.options['horizon']
          del acado.options['Tmax']
          acado.debug(False)
          acado.iter                = 80
          acado.options['steps']    = 20

     elif label == "refine":
          acado.iter                = 80
          acado.options['steps']    = 20

class ConnectAcado(ConnectAbstract):
     def __init__(self,acado):
          self.acado = acado
          if 'availableJobs' not in acado.__dict__: acado.setup_async()
          self.threshold = 1e-3
          self.idxBest   = None   # Store the index of the best previous trial

     def __call__(self,x1,x2,verbose=False):
          dq     = (x2[:2] - x1[:2])%(2*np.pi)
          x2     = x2.copy()
          x2[:2] = x1[:2] + dq
          PI0    = zero(4); PI0[0] = -2*np.pi
          PI1    = zero(4); PI1[1] = -2*np.pi

          trials = [ x2, x2+PI0, x2+PI1, x2+PI0+PI1 ]
          scores = [ np.inf, ]*4
          jobs   = {}
          
          for i,x2 in enumerate(trials):
               if verbose: print 'Trying (%d)'%i,x2.T
               jobs[i] = self.acado.run_async(x1,x2)

          idxBest = -1; scoreBest = np.inf
          for i,x2 in enumerate(trials):
               if self.acado.join(jobs[i],x1=x1,x2=x2,threshold=self.threshold):
                    cost = self.acado.opttime(jobs[i])
                    if cost < scoreBest: idxBest = jobs[i]; scoreBest = cost

          self.idxBest = idxBest
          return idxBest>=0

     def states(self):
          return self.acado.states(self.idxBest)
     def controls(self): 
          return self.acado.controls(self.idxBest)
     def cost(self):
          return self.acado.opttime(self.idxBest)
     def times(self):
          return self.acado.times(self.idxBest)

class PendulumStateDiff:
     def __init__(self,NQ):
          self.NQ = NQ

     def __call__(self,x1,x2):
          '''return dx modulo 2*PI such that x2 = x1 + dx, dx minimal'''
          PI = np.pi;     PPI = 2*PI
          dx = x2-x1
          dq = (dx[:self.NQ]+PI)%PPI - PI
          dv =  dx[self.NQ:]
          return np.concatenate([dq,dv])

# --- MAIN -----------------------------------------------------------------------
# --- MAIN -----------------------------------------------------------------------
# --- MAIN -----------------------------------------------------------------------

'''
EXTEND_PRM   = 6
LOAD_PRM     = False
SAMPLE_GRID  = True
REFINE_GRID  = 5
'''
EXTEND_PRM   = 0
LOAD_PRM     = True
SAMPLE_GRID  = False
REFINE_GRID  = 0


RANDOM_SEED = 999 # int((time.time()%10)*1000)
print "Seed = %d" %  RANDOM_SEED
np .random.seed     (RANDOM_SEED)
random.seed         (RANDOM_SEED)

env                 = Pendulum(2,length=.5,mass=3.0,armature=.2,withDisplay=False)
env.withSinCos      = False             # State is dim-3: (cosq,sinq,qdot) ...
NX                  = env.nobs          # ... training converges with q,qdot with 2x more neurones.
NU                  = env.nu            # Control is dim-1: joint torque

env.vmax            = 100.
env.Kf              = np.diagflat([ 0.2, 2. ])
env.modulo          = False
env.DT              = 0.15
env.NDT             = 1
env.umax            = np.matrix([5.,10.]).T

env.qlow[1]         = -np.pi
env.qup [1]         = +np.pi


acado = AcadoConnect(acadoPath,
                     model=env.model,
                     datadir=acadoTxtPath)
config(acado,'connect',env)

# --- PRM ---
# --- PRM ---
# --- PRM ---
prm = PRM(GraphPendulumDouble(),
          sampler = env.reset,
          checker = lambda x:True,
          nearestNeighbor = NearestNeighbor(DistanceSO3([1,.1])),
          connect = ConnectAcado(acado))

if LOAD_PRM:
     prm.graph.load(dataRootPath)

if EXTEND_PRM>5:
     for i in range(5):
          simplePrm(20,10,10,True)
          print 'Sleeping 10 ... it is time for a little CTRL-C ',time.ctime()
          time.sleep(10)
     graph.save(dataRootPath+'_100pts')

if EXTEND_PRM>4:
     ### Filling the prm with additional points at low speed.
     env.vup[:] = 3.
     env.vlow[:] = -3.
     for i in range(5):
          simplePrm(10,50,50,False)
          print 'Sleeping 10 ... it is time for a little CTRL-C ',time.ctime()
          time.sleep(10)
     graph.save(dataRootPath+'_200pts')

if EXTEND_PRM>3:
     ### Filling the prm with additional points close to up equilibrium.
     env.qup[:] = .2
     env.qlow[:] = -.2
     env.vup[:] = .5
     env.vlow[:] = -.5
     prevSize = len(graph.x)
     for i in range(5):
          simplePrm(10,20,20,False)
          print 'Sleeping 10 ... it is time for a little CTRL-C ',time.ctime()
          time.sleep(10)
     graph.save(dataRootPath+'_400pts')

if EXTEND_PRM>2:
     ### Filling the prm with additional points close to joint limit.
     env.qlow = np.matrix([-5, -np.pi]).T
     env.qup  = np.matrix([ 5, -.6*np.pi]).T
     env.vlow[:] = -.5
     env.vup[:] = .5
     prevSize = len(graph.x)
     for i in range(5):
          simplePrm(10,20,20,False)
          print 'Sleeping 10 ... it is time for a little CTRL-C ',time.ctime()
          time.sleep(10)

if EXTEND_PRM>1:
     env.qlow = np.matrix([-5,  .6*np.pi]).T
     env.qup  = np.matrix([ 5,     np.pi]).T
     env.vlow[:] = -.5
     env.vup[:] = .5
     prevSize = len(graph.x)
     for i in range(5):
          simplePrm(10,20,20,False)
          print 'Sleeping 10 ... it is time for a little CTRL-C ',time.ctime()
          time.sleep(10)

if EXTEND_PRM>0:
     print 'Connect all points to zero (at least tries)',time.ctime()
     connectToZero(graph)
     print 'Densify PRM',time.ctime()
     densifyPrm(graph)
     connexifyPrm(graph)

     prm.graph.save(dataRootPath)

# --- GRID ---
# --- GRID ---
# --- GRID ---

oprm = OptimalPRM.makeFromPRM(prm,acado=prm.connect.acado,stateDiff=PendulumStateDiff(2))
grid = GridPolicy(oprm)
EPS = 1e-3
grid.setGrid([ -np.pi,-np.pi+EPS,0,0],[np.pi,np.pi-EPS,EPS,EPS],1.)

if SAMPLE_GRID:     
     print 'Sample the grid',time.ctime()
     grid.sample()
else:
     grid.load(dataRootPath+'/grid.npy')

if REFINE_GRID>3:
     print 'Fill the grid',time.ctime()
     refineGrid(data,NNEIGHBOR=30,PERCENTAGE=.9, 
                RANDQUEUE=[ i for i,d in enumerate(data) if d.cost>100])
     refineGrid(data,NNEIGHBOR=100,PERCENTAGE=.9, 
                RANDQUEUE=[ i for i,d in enumerate(data) if d.cost>100])
     np.save(dataRootPath+'/grid_filled.npy',data)

if REFINE_GRID>3:
     print 'Refine the grid',time.ctime()
     refineGrid(data,5000,NNEIGHBOR=20,PERCENTAGE=.9)
     np.save(dataRootPath+'/grid.npy',data)

if REFINE_GRID>2:
     print 'Refine outliers in the grid',time.ctime()
     refineGrid(data,5000,NNEIGHBOR=30,PERCENTAGE=.8,RANDQUEUE=[ i for i,d in enumerate(data) if d.cost>3 ])
     np.save(dataRootPath+'/grid.npy',data)

if REFINE_GRID>2:
     print 'Refine outliers in the grid',time.ctime()
     refineGrid(data,5000,NNEIGHBOR=20,PERCENTAGE=1.1)
     np.save(dataRootPath+'/grid.npy',data)

# --- MISC ---
# --- MISC ---
# --- MISC ---

fromcursor = FromCursor(oprm,grid,env)
acado = oprm.acado
nearest = prm.nearestNeighbor
plt.ion()
