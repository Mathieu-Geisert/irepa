from pinocchio.utils import *
import pinocchio as se3
from time import sleep
from numpy.linalg import inv,norm
import math
import time
import heapq
import random
from pendulum import Pendulum
import matplotlib.pylab as plt

RANDOM_SEED = 9 # int((time.time()%10)*1000)
print "Seed = %d" %  RANDOM_SEED
np .random.seed     (RANDOM_SEED)
random.seed         (RANDOM_SEED)

env                 = Pendulum(1,withDisplay=False)       # Continuous pendulum
env.withSinCos      = False             # State is dim-3: (cosq,sinq,qdot) ...
NX                  = env.nobs          # ... training converges with q,qdot with 2x more neurones.
NU                  = env.nu            # Control is dim-1: joint torque

env.vmax            = 100.
env.Kf              = 0.2
env.modulo          = False

env.DT              = 0.15
env.NDT             = 1
NSTEPS              = 32

# Shortcut function to convert SE3 to 7-dof vector.
M2gv      = lambda M: XYZQUATToViewerConfiguration(se3ToXYZQUAT(M))
def place(objectId,M):
     robot.viewer.gui.applyConfiguration(objectId, M2gv(M))
     robot.viewer.gui.refresh() # Refresh the window.

################################################################################
################################################################################
################################################################################


class Graph:
     def __init__(self):
          self.x        = []
          self.children = {}
          self.connex   = []    # ID of the connex component the node is belonging to
          self.nconnex = 0      # number of connex components
          self.existingConnex = [] # List of existing connex component ID
          self.localPath = {}
          self.localControl = {}
          self.edgeCost = {}

     def addNode(self, x=None, newConnex=False):
          '''
          Create the memory to store a new edge. Initialize all components to None.
          Create an empty list of children.
          '''
          idx = len(self.children)
          self.children[idx] = []
          self.x.append(x)
          self.connex.append(None)
          if newConnex: self.newConnex(idx)

          return idx

     def addEdge(self,first,second,orientation = 0,
                 localPath = None, localControl = None, cost = -1 ):
          '''
          Add edge from first to second. Also add edge from second to first if orientation
          is null.
          '''
          assert( first in self.children and second in self.children )

          self.children[first].append(second)
          self.localPath[ first,second ]      = localPath
          self.localControl[ first,second ]   = localControl
          self.edgeCost[ first,second ]       = cost

          if localPath is not None: 
               x1 = self.x[first]
               x2 = self.x[second]
               plt.figure(1)
               plt.plot([x1[0,0],x2[0,0]], [x1[1,0],x2[1,0] ])
               plt.axis([env.qlow,env.qup,env.vlow,env.vup])
               plt.draw()
               plt.figure(2)
               plt.plot( localPath[:,0], localPath[:,1] )
               plt.axis([env.qlow,env.qup,env.vlow,env.vup])
               plt.draw()

     def newConnex(self,idx):
          '''
          Create a new connex component for node <idx>
          '''
          self.connex[idx] = self.nconnex
          self.existingConnex.append(self.nconnex)
          self.nconnex += 1
     def renameConnex(self,past,future):
          '''
          Change the index of the all the nodes belonging to a connex component.
          Useful when merging two connex components.
          '''
          try:
               self.existingConnex.remove(past)
               self.connex = [ c if c!=past else future for c in self.connex ]
          except:
               pass
     def connexIndexes(self,connex):
          '''Return the list of all node indexes belonging to connex component <connex>.'''
          return [ i for i,c in enumerate(self.connex) if c == connex ]
          
     def removeEdge(self,idx,idx2):
          self.children[idx].remove(idx2)
          del self.localPath[idx,idx2]
          del self.localControl[idx,idx2]
          del self.edgeCost[idx,idx2]

     def removeNode(self,idx):
          for idx2 in reversed(self.children[idx]):
               self.removeEdge(idx,idx2)
          for idx2 in [ k for k,v in self.children.items() if idx in v ]:
               self.removeEdge(idx2,idx)
          del self.children[idx]
          self.x[idx] = None
          self.connex[idx] = None

     def removeConnex(self,ic):
          for idx in [ n for n,c in enumerate(self.connex) if c==ic ]:
               self.removeNode(idx)

     def plot(self,withPath=True):
          colorcycle = plt.subplots()[1]._get_lines.color_cycle
          colors = [ colorcycle.next() for _ in range(1000) ]

          for x in [ x for x in self.x if x is not None ]:
               plt.plot(x[0,0],x[1,0],'k+')
          for (i0,i1),path in self.localPath.items():
               c = self.connex[i0]
               if withPath: plt.plot(path[:,0],path[:,1],colors[c])
               else:        plt.plot(path[[0,-1],0],path[[0,-1],1],colors[c])

     def save(self,path):
          if path[-1]!='/': path+='/'
          for s in [ 'x', 'children', 'connex', 'localPath', 'localControl', 'edgeCost' ]:
               np.save(path+s+'.npy',self.__dict__[s])
     def load(self,path):
          if path[-1]!='/': path+='/'
          for s in [ 'x', 'children', 'connex', 'localPath', 'localControl', 'edgeCost' ]:
               self.__dict__[s] = np.load(path+s+'.npy')[()]
          self.nconnex = max(self.connex)+1
          self.existingConnex = sorted(list(set(self.connex)))



def check(x):
     ''' 
     Check that the robot constraints are satisfied. Return True if satisfied, False otherwise.
     '''
     return True
     
def nearestNeighbor(x1, xs, nneighbor = 1, hdistance = lambda x1,x2: norm(x1-x2)  ):
     '''
     Return the indexes <nneighbor> nearest neighbors of new configuration x1.
     x1 is the central node from which distances are computed.
     xs is the list of configuration to search in.
     nneighbor is the number of closest neighbors to return.
     condition is a function of node to filter (ex use it to select only the nodes of 
     one component of the graph).
     <hdistance> defines  the (heuristic) distance function to be used for nearest neighbor algorithm.
     '''
     if len(xs) <= nneighbor: return range(len(xs))
     return np.argpartition([ hdistance(x1,x2) for x2 in xs],
                            nneighbor)[:nneighbor]

from acado_connect import AcadoConnect

class ConnectAcado:
     def __init__(self):
          acado = self.acado = AcadoConnect(model=env.model)
          acado.setTimeInterval(1.0)
          acado.options['steps']    = 10
          acado.options['shift']    = 0
          acado.options['iter']     = 20
          acado.options['friction'] = 0.2
          acado.options['printlevel']=1   # To get error code
          acado.setTimeInterval(1.0)

          self.threshold = 1e-3

     def __call__(self,x1,x2):
          try:
               self.acado.run(x1,x2)
          except:
               return False

          print self.acado.retcode,x1.T,x2.T
          if self.acado.retcode == 0: return True
          X = self.acado.states()
          return norm(x1-X[0,:]) < self.threshold \
              and norm(x2-X[-1,:]) < self.threshold
                                
connect = ConnectAcado()

def simplePrm(nsamples = 1000):
     NCONNECT = 3       # Number of nearest neighbors to try to connect with.
     for _ in range(nsamples):
          x = env.reset()
          if not check(x): continue
          idx = graph.addNode()                         # Index of the new node
          graph.x[idx] = x                              # Add a new node for configuration x
          connected = False
          for idx2 in nearestNeighbor(x,graph.x[:-1],NCONNECT):
               if connect(x,graph.x[idx2]):              # Try connect x to new neighbors
                    graph.addEdge(idx,idx2,+1,           # Add a new edge
                                  localPath = connect.acado.states(),
                                  localControl = connect.acado.controls() )
                    connected = True
          if idx>0 and not connected:
               graph.x = graph.x[:-1]
               del graph.children[idx]

plt.ion()
graph = Graph()
graph.addNode(newConnex=True)
graph.x[0] = zero(2)

nsamples = 200
NCONNECT = 20
NBEST   = 2

connect.acado.setTimeInterval(.2)

acado=connect.acado

for _ in range(nsamples):
     x = env.reset()
     if not check(x): continue
     idx = graph.addNode(newConnex=True)           # Index of the new node
     graph.x[idx] = x                              # Add a new node for configuration x
     edges = []
     for idx2 in nearestNeighbor(x,graph.x[:-1],NCONNECT):
          if connect(x,graph.x[idx2]):              # Try connect x to new neighbors
               edges.append( [ idx2, connect.acado.opttime(), 
                               connect.acado.states(),
                               connect.acado.controls() ] )
     for idx2, cost, X,U in sorted(edges, key = lambda e: e[1])[:min(len(edges),NBEST)]:
          graph.addEdge(idx,idx2,+1,           # Add a new edge
                        localPath = X,
                        localControl = U,
                        cost = cost)
          graph.renameConnex(graph.connex[idx2],graph.connex[idx])


# count = [ 0 for _ in graph.connex ]
# for k in graph.connex: count[k] += 1
# for ic in [ k for k,c in enumerate(count) if c<6 and c>0]: graph.removeConnex(ic)


#def plotmodulo(x,y, **kwargs):
     
     



