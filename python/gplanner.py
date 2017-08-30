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

from specpath import acadoBinDir,acadoTxtPath
from bicopter import Bicopter
from acado_connect import AcadoConnect

dataRootPath = 'data/planner/bicopter'
     
# --- PENDULUM DOUBLE ------------------------------------------------------------
# --- PENDULUM DOUBLE ------------------------------------------------------------
# --- PENDULUM DOUBLE ------------------------------------------------------------

class GraphBicopter(Graph):
     '''Specialization of Graph to have the display dedicated to double pendulum.'''
     def __init__(self): Graph.__init__(self)

     def plotState(self,q,v=None,color='r',marker='o',lseg=.1):
          q = q.flat
          plt.plot([q[0]+np.cos(q[2])*lseg,q[0]-np.cos(q[2])*lseg],
                   [q[1]+np.sin(q[2])*lseg,q[1]-np.sin(q[2])*lseg],
                   linestyle='-',marker=marker,linewidth=1,color=color)

     def plotNode(self,idx,plotstr='k',**kwargs):
          x = self.x[idx]
          if x is None: return
          self.plotState(x,color=plotstr)
          plt.plot(x[0,0],x[1,0],plotstr,**kwargs)

     def plotEdge(self,idx1,idx2,plotstr='k',withTruePath = False,**kwargs):
          path = self.states[idx1,idx2]
          if withTruePath: 
               plt.plot(path[:,0],path[:,1],plotstr,**kwargs)
          else: 
               plt.plot(path[[0,-1],0],path[[0,-1],1],plotstr,**kwargs)

def config(acado,label,env=None):
     del acado.options['icontrol']
     if env is not None:
          acado.options['umax']     = "%.2f %.2f" % tuple([x for x in env.umax])
          
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

     def __call__(self,x1,x2,verbose=False):
          try: self.acado.run(x1,x2)
          except: return False
          return True

     def states(self):
          return self.acado.states()
     def controls(self): 
          return self.acado.controls()
     def cost(self):
          return self.acado.opttime()
     def times(self):
          return self.acado.times()

class BicopterStateDiff:
     def __call__(self,x1,x2):
          return x2-x1

# --- MAIN -----------------------------------------------------------------------
# --- MAIN -----------------------------------------------------------------------
# --- MAIN -----------------------------------------------------------------------

EXTEND_PRM   = [1,2,6]
LOAD_PRM     = False
LOAD_GRID    = False
SAMPLE_GRID  = False
REFINE_GRID  = []
'''
EXTEND_PRM   = 0
LOAD_PRM     = True
SAMPLE_GRID  = False
REFINE_GRID  = 0
'''


RANDOM_SEED = 999 # int((time.time()%10)*1000)
print "Seed = %d" %  RANDOM_SEED
np .random.seed     (RANDOM_SEED)
random.seed         (RANDOM_SEED)

plt.ion()

env     = Bicopter(withDisplay=True)
NX      = 6
NQ      = 3
NV      = 3

acado = AcadoConnect(acadoBinDir+"connect_bicopter",
                     datadir=acadoTxtPath)
acado.NQ = NQ
acado.NV = NV
config(acado,'connect',env)


# --- PRM ---
# --- PRM ---
# --- PRM ---
prm = PRM(GraphBicopter(),
          sampler = env.reset,
          checker = lambda x:True,
          nearestNeighbor = NearestNeighbor(DistanceSO3([1,.1])),
          connect = ConnectAcado(acado))

prm.graph.addNode(newConnex=True)
prm.graph.x[0] = zero(NX)

connect = prm.connect
nearest = prm.nearestNeighbor
graph   = prm.graph


x0 = zero(6)
x1 = zero(6); x1[0] = .1

acado.run(x0,x1)

if LOAD_PRM:
     prm.graph.load(dataRootPath)

if 1 in EXTEND_PRM:
     print '### Initial sampling of PRM',time.ctime()
     for i in range(5):
          prm(20,10,10,True)
          print 'Sleeping 10 ... it is time for a little CTRL-C ',time.ctime()
          #time.sleep(1)
     graph.save(dataRootPath+'_100pts')

if 2 in EXTEND_PRM:
     print '### Filling the prm with additional points at low speed.',time.ctime()
     env.vup[:] = .2
     env.vlow[:] = -.2
     for i in range(5):
          prm(10,50,50,False)
          print 'Sleeping 10 ... it is time for a little CTRL-C ',time.ctime()
          time.sleep(10)
     graph.save(dataRootPath+'_200pts')

if 3 in EXTEND_PRM:
     print '### Filling the prm with additional points close to up equilibrium.',time.ctime()
     env.qup[:] = .2
     env.qlow[:] = -.2
     env.vup[:] = .5
     env.vlow[:] = -.5
     prevSize = len(graph.x)
     for i in range(5):
          prm(10,20,20,False)
          print 'Sleeping 10 ... it is time for a little CTRL-C ',time.ctime()
          time.sleep(10)
     graph.save(dataRootPath+'_400pts')

if 4 in EXTEND_PRM:
     print '### Filling the prm with additional points close to joint limit.',time.ctime()
     env.qlow = np.matrix([-5, -np.pi]).T
     env.qup  = np.matrix([ 5, -.6*np.pi]).T
     env.vlow[:] = -.5
     env.vup[:] = .5
     prevSize = len(graph.x)
     for i in range(5):
          prm(10,20,20,False)
          print 'Sleeping 10 ... it is time for a little CTRL-C ',time.ctime()
          time.sleep(10)

if 6 in EXTEND_PRM:
     print '### Connect all points to zero (at least tries)',time.ctime()
     connectToZero(graph)
     print 'Densify PRM',time.ctime()
     densifyPrm(graph)
     connexifyPrm(graph)

     prm.graph.save(dataRootPath)

# --- GRID ---
# --- GRID ---
# --- GRID ---

oprm = OptimalPRM.makeFromPRM(prm,acado=prm.connect.acado,stateDiff=BicopterStateDiff())
grid = GridPolicy(oprm)
EPS = 1e-3
grid.setGrid([ -np.pi,-np.pi+EPS,0,0],[np.pi,np.pi-EPS,EPS,EPS],1.)

if SAMPLE_GRID:     
     print 'Sample the grid',time.ctime()
     grid.sample()
else:
     grid.load(dataRootPath+'/grid.npy')

if 1 in REFINE_GRID:
     print 'Fill the grid',time.ctime()
     refineGrid(data,NNEIGHBOR=30,PERCENTAGE=.9, 
                RANDQUEUE=[ i for i,d in enumerate(data) if d.cost>100])
     refineGrid(data,NNEIGHBOR=100,PERCENTAGE=.9, 
                RANDQUEUE=[ i for i,d in enumerate(data) if d.cost>100])
     np.save(dataRootPath+'/grid_filled.npy',data)

if 2 in REFINE_GRID:
     print 'Refine the grid',time.ctime()
     refineGrid(data,5000,NNEIGHBOR=20,PERCENTAGE=.9)
     np.save(dataRootPath+'/grid.npy',data)

if 3 in REFINE_GRID:
     print 'Refine outliers in the grid',time.ctime()
     refineGrid(data,5000,NNEIGHBOR=30,PERCENTAGE=.8,RANDQUEUE=[ i for i,d in enumerate(data) if d.cost>3 ])
     np.save(dataRootPath+'/grid.npy',data)

if 4 in REFINE_GRID:
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


def dg(**kw):
     for k,v in kw.items(): print k,v
