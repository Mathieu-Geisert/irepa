from pinocchio.utils import *
import pinocchio as se3
from time import sleep
from numpy.linalg import inv,norm
import math
import time
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
               # plt.figure(1)
               # plt.plot([x1[0,0],x2[0,0]], [x1[1,0],x2[1,0] ])
               # plt.axis([env.qlow,env.qup,env.vlow,env.vup])
               # plt.draw()
               # plt.figure(2)
               # plt.plot( localPath[:,0], localPath[:,1] )
               # plt.axis([env.qlow,env.qup,env.vlow,env.vup])
               # plt.draw()

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
          if past==future: return
          if past in self.existingConnex:        self.existingConnex.remove(past)
          self.connex = [ c if c!=past else future for c in self.connex ]
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
          if idx == len(self.x)-1: 
               self.x = self.x[:-1]
               self.connex = self.connex[:-1]

     def removeConnex(self,ic):
          for idx in [ n for n,c in enumerate(self.connex) if c==ic ]:
               self.removeNode(idx)

     def plotNode(self,idx,plotstr='k+',**kwargs):
          x = self.x[idx]
          if x is None: return
          plt.plot(x[0,0],x[1,0],plotstr,**kwargs)
          plt.plot(x[0,0]+np.pi*2,x[1,0],plotstr,**kwargs)
          plt.plot(x[0,0]-np.pi*2,x[1,0],plotstr,**kwargs)
     def plotEdge(self,idx1,idx2,plotstr='k',withTruePath = False,**kwargs):
          path = self.localPath[idx1,idx2]
          if withTruePath: 
               plt.plot(path[:,0],path[:,1],plotstr,**kwargs)
               plt.plot(path[:,0]-2*np.pi,path[:,1],plotstr,**kwargs)
               plt.plot(path[:,0]+2*np.pi,path[:,1],plotstr,**kwargs)
          else: 
               plt.plot(path[[0,-1],0],path[[0,-1],1],plotstr,**kwargs)
               plt.plot(path[[0,-1],0]+2*np.pi,path[[0,-1],1],plotstr,**kwargs)
               plt.plot(path[[0,-1],0]-2*np.pi,path[[0,-1],1],plotstr,**kwargs)
          
     def plot(self,withPath=True):
          colorcycle = plt.gca()._get_lines.color_cycle
          colors = [ colorcycle.next() for _ in range(1000) ] \
              if len(self.existingConnex)>1 else ['k','.65']*1000
          

          for ix in range(len(self.x)): self.plotNode(ix)
          for (i0,i1),path in self.localPath.items():
               c = 2*self.connex[i0] + int(i0<i1)
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

     # --- HELPERS ---
     def connexComponents(self):
          cc = dict()
          for n,c in enumerate(self.connex):
               if c not in cc: cc[c] = []
               cc[c].append(n)
          return cc
     def descendants(self,node):
          sons = set([node])
          nsons = 0
          while nsons!=len(sons):
               nsons = len(sons)
               for n in list(sons): sons |= set(self.children[n])
          return sons
     def subgraph(self,nodes):
          '''Return all edges connecting nodes'''
          edges = []
          for n,cs in [(n,cs) for n,cs in self.children.items() if n in nodes]:
               for c in [c for c in cs if c in nodes]:
                    edges.append( [n,c] )
          return edges


def check(x):
     ''' 
     Check that the robot constraints are satisfied. Return True if satisfied, False otherwise.
     '''
     return True
     
class WeightedNorm:
     def __init__(self,weights):
          self.weights = weights
     def __call__(self,x1,x2):
          return np.sqrt(np.sum(self.weights*np.square(x1-x2)))

class SO3Norm:
     def __init__(self,weights):
          self.weights = weights
     def __call__(self,x1,x2):
          cs1 = np.array([ np.cos(x1.flat[0]), np.sin(x1.flat[0]), x1.flat[1] ])
          cs2 = np.array([ np.cos(x2.flat[0]), np.sin(x2.flat[0]), x2.flat[1] ])
          return np.sqrt(np.sum(self.weights*np.square(cs1-cs2)))

class NearestNeighbor:
     def __init__(self,hdistance = lambda x1,x2: norm(x1-x2)):
          self.hdistance = hdistance
     def __call__(self,x1, xs, nneighbor = 1, hdistance = None):
          if hdistance is None: hdistance = self.hdistance
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

#wnorm = WeightedNorm([1,.4])
wnorm = SO3Norm([1,1,.4])
nearestNeighbor = NearestNeighbor(wnorm)

from acado_connect import AcadoConnect

class ConnectAcado:
     def __init__(self):
          acado = self.acado = AcadoConnect(model=env.model)
          acado.setTimeInterval(1.0)
          acado.options['steps']    = 10
          acado.options['shift']    = 0
          acado.options['iter']     = 20
          acado.options['friction'] = 0.2
          acado.setTimeInterval(1.0)

          self.threshold = 1e-3

     def __call__(self,x1,x2):
          try:
               self.acado.run(x1,x2)
          except:
               # try:
               #      self.acado.run(x1-np.matrix([2*np.pi,0]).T,x2)
               # except:
               return False

          #print self.acado.retcode,x1.T,x2.T
          if self.acado.retcode == 0: return True
          X = self.acado.states()
          return norm(x1-X[0,:]) < self.threshold \
              and norm(x2-X[-1,:]) < self.threshold

                                
connect = ConnectAcado()

XLOW = np.matrix

def simplePrm(NSAMPLES = 1000, NCONNECT = 20, NBEST = 3, interactivePlot = False):
     '''
     Basic PRM algorithm: sample <nsamples> points and try to connect each to <nconnect>
     neighbor. Minor variation: keep only the best <nbest> connections.
     nconnect: Number of nearest neighbors to try to connect with.
     '''
     connect.acado.debug(False) # just in case
     for _ in range(NSAMPLES):
          print "PRM sample #",len(graph.x)
          x = env.reset()
          if not check(x): continue
          idx = graph.addNode(newConnex=True)           # Index of the new node
          graph.x[idx] = x                              # Add a new node for configuration x
          edges = []
          doConnect = False
          MODULO = np.matrix([-np.sign(x[0,0])*2*np.pi,0]).T
          for idx2 in nearestNeighbor(x,graph.x[:-1],NCONNECT):
               # From x to graph
               if connect(x,graph.x[idx2]) or connect(x+MODULO,graph.x[idx2]):
                    edges.append( [ idx, idx2, connect.acado.opttime(), 
                                    connect.acado.states(),
                                    connect.acado.controls() ] )
                    doConnect = True
               # From graph to x
               if connect(graph.x[idx2],x) or connect(graph.x[idx2],x+MODULO):
                    edges.append( [ idx2, idx, connect.acado.opttime(), 
                                    connect.acado.states(),
                                    connect.acado.controls() ] )
          '''
          if not doConnect:  # Force tree structure by cosntraining upward connectivity to 0
               graph.removeNode(idx)  
               continue
               '''
          if interactivePlot: graph.plotNode(idx)
          for idx1, idx2, cost, X,U in sorted(edges, key = lambda e: e[1])[:min(len(edges),NBEST)]:
               graph.addEdge(idx1,idx2,+1,           # Add a new edge
                             localPath = X,
                             localControl = U,
                             cost = cost)
               graph.renameConnex(graph.connex[idx2],graph.connex[idx1])
               if interactivePlot: graph.plotEdge(idx1,idx2,withTruePath=True,
                                                  color= 'k' if idx1<idx2 else '.5')
          if interactivePlot: plt.draw() #; time.sleep(1.)

def visibilityPrm(NSAMPLES = 1000, NCONNECT = 20, interactivePlot = False):
     '''
     Basic PRM algorithm: sample <nsamples> points and try to connect each to <nconnect>
     neighbor. Minor variation: keep only the best <nbest> connections.
     nconnect: Number of nearest neighbors to try to connect with.
     '''
     for _ in range(NSAMPLES):
          x = env.reset()
          if not check(x): continue
          idx = graph.addNode(newConnex=True)           # Index of the new node
          graph.x[idx] = x                              # Add a new node for configuration x
          edges = {}

          for connex in graph.existingConnex:
               edges[connex] = { 'way': [], 'back': [] }
               iconnex = graph.connexIndexes(connex)               # Indexes of current connex component.
               for idx2 in nearestNeighbor(x,[ graph.x[i] for i in iconnex]):
                    # Try connect from x to graph
                    if connect(x,graph.x[iconnex[idx2]]):
                         edges[connex]['way'].append( [ idx, iconnex[idx2], connect.acado.opttime(), 
                                                        connect.acado.states(),
                                                        connect.acado.controls() ] )
                    # From graph to x
                    if connect(graph.x[idx2],x):
                         edges[connex]['back'].append( [ iconnect[idx2], idx, connect.acado.opttime(), 
                                                         connect.acado.states(),
                                                         connect.acado.controls() ] )
               if len(edges[connex]['way'])==0 or len(edges[connex]['back'])==0:
                    del edges[connex,'way']
                    del edges[connex,'back']

          if len(edges) == 1 : 
               print 'Connect only once ... cancel'
               graph.removeNode(idx)  
               continue

          if interactivePlot: graph.plotNode(idx)

          for connex,wayback in edges.items():
               for es in wayback.values:
                    for idx1, idx2, cost, X,U in sorted(es, key = lambda e: e[1])[:min(len(edges),NBEST)]:
                         graph.addEdge(idx1,idx2,+1,           # Add a new edge
                                       localPath = X,
                                       localControl = U,
                                       cost = cost)
               graph.renameConnex(graph.connex[idx2],graph.connex[idx1])
               if interactivePlot: graph.plotEdge(idx1,idx2,withTruePath=True)
          if interactivePlot: plt.draw() #; time.sleep(1.)


def prunePRM(graph,percent=.8):
     '''
     Tentative to reduce the graph while keeping information ... seems more difficult
     to achieve than this naive trial.
     '''
     nedge = len(graph.edgeCost)
     for [idx1,idx2],cost in sorted(graph.edgeCost.items(),key=lambda icost:icost[1])[int(nedge*percent):]:
          graph.removeEdge(idx1,idx2)
     
     NEST = 100
     dist = []
     for i in range(NEST):
          i1 = random.randint(0,len(graph.x)-1)
          i2 = random.randint(0,len(graph.x)-1)
          if i1 == i2: continue
          if graph.x[i1] is None or graph.x[i2] is None: continue
          dist.append(norm(graph.x[i1]-graph.x[i2]))

     plt.figure(99)
     dist = sorted(dist)
     THR = dist[int(NEST*(1-percent)*.2)] # Threshold
     print "Dist from %.2f to %.2f: prune to %.2f" % (dist[0],dist[-1],THR)
     graph.rm = []
     for i in range(NEST):
          i1 = random.randint(0,len(graph.x)-1)
          i2 = random.randint(0,len(graph.x)-1)
          if i1 == i2: continue
          if graph.x[i1] is None or graph.x[i2] is None: continue
          dist = norm(graph.x[i1]-graph.x[i2])
          if dist < THR:
               x1 = graph.x[i1]
               x2 = graph.x[i2]
               plt.plot(x1[0,0],x1[1,0],'r+',markeredgewidth=5)
               plt.plot(x2[0,0],x2[1,0],'b+',markeredgewidth=5)
               plt.plot([ x1[0,0],x2[0,0] ],
                        [ x1[1,0],x2[1,0] ],'k')
               graph.rm.append(graph.x[i1])
               graph.removeNode(i1)

          


          



plt.ion()

graph = Graph()
graph.addNode(newConnex=True)
graph.x[0] = zero(2)
# nsamples = 200
# NCONNECT = 20
# NBEST   = 2



connect.acado.setTimeInterval(.5)

acado=connect.acado
acado.options['printlevel']=1

#D=np.load('data/databasexx.np')
D=np.load('data/netdata.npy')
plt.scatter(D[:,0].flat,D[:,1].flat,c=D[:,3].flat,linewidths=0,s=50,alpha=.8)
plt.scatter((D[:,0]-np.pi*2).flat,D[:,1].flat,c=D[:,3].flat,linewidths=0,s=50,alpha=.8)
plt.scatter((D[:,0]+np.pi*2).flat,D[:,1].flat,c=D[:,3].flat,linewidths=0,s=50,alpha=.8)
plt.axis([env.qlow*1.5,env.qup*1.5,env.vlow*1.2,env.vup*1.2])

#simplePrm(3,1000,1000,True)

#simplePrm(10,1000,1000,True)
#simplePrm(50,50,5,True)
#simplePrm(50,50,5,True)


#simplePrm(15,20,3,True)
#simplePrm(200,200,5,True)
graph.load('data/')



from astar import astar
def pathFrom(idx):
     if 0 not in graph.descendants(idx):
          print "0 not in descendance of idx:",graph.descendants(idx)
          raise graph.descendants(idx)

     traj = astar(graph,idx,0)
     prev = traj[0]
     path = []
     control = []
     times = []
     time = 0
     x = graph.x[idx]

     # q1 = q2 - mod2pi(q1,q2)
     mod2Pi = lambda x1,x2: np.array([ round((x2.flat[0]-x1.flat[0])/2/np.pi)*2*np.pi, 0 ])

     for cur in traj[1:]:
          T = graph.edgeCost[prev,cur]
          N = graph.localPath[prev,cur].shape[0]-1
          local = graph.localPath[prev,cur].copy()
          q0 = local[0,0]
          q  = x[0,0]

          #print prev,cur,q,q0
          #assert( (q0%np.pi)-(q%np.pi)<1e-2 )
          #mod = round((q0-q)/2/np.pi)
          #q0 -= mod*2*np.pi
          
          #local[:,0] -= mod*2*np.pi
          local -= mod2Pi(x,local[0,:])

          path   .append( local[:-1,:] )
          control.append( graph.localControl[prev,cur][:-1])
          times  .append( np.arange(0.,N)/N * T + time)
          time += T

          prev = cur
          x = local[-1:,:]

     path.append( graph.x[0].T - mod2Pi(x,graph.x[0]) )
     control.append(zero(1))
     times.append(time)

     times = np.hstack(times)
     X = np.vstack(path)
     U = np.vstack(control)
     return X,U,times

def optpathFrom(idx):
     X,U,times = pathFrom(idx)
     time = times[-1]

     np.savetxt(acado.options['istate'],   np.vstack([times/time,X.T]).T )
     np.savetxt(acado.options['icontrol'], np.vstack([times/time,U.T]).T )
     acado.options['horizon'] = time
     acado.options['Tmax'] = 4*time

     #x0,x1 = graph.x[idx],graph.x[0]
     x0,x1 = X[0,:],X[-1,:]
     #acado.debug()
     acado.iter = 100
     acado.options['steps'] = 50

     acado.run( x0,x1,autoInit=False)

     Xac = acado.states()
     Uac = acado.controls()

     return Xac,Uac,[]

def test(idx = None):
     if idx is None: idx = random.randint(0,len(graph.x)-1)
     print "Opt Path with idx = ",idx
     X,_,_ = pathFrom(idx)
     Xac,_,_ = optpathFrom(idx)
     plt.plot(X[:,0],X[:,1],'r', linewidth=2)
     plt.plot(Xac[:,0],Xac[:,1],'g', linewidth=2)



def settest(idx2=0):
     x1=graph.x[-1].copy()
     MODULO = np.matrix([-np.sign(x1[0,0])*2*np.pi,0]).T
     x2=graph.x[idx2].copy()
     globals()['idx1'] = -1
     globals()['x1'] = x1
     globals()['MODULO'] = MODULO
     globals()['x2'] = x2
     globals()['idx2'] = idx2


from capture_cursor import *
cursorCoordinate.connect()

def nodeFromCursor():
     x = cursorCoordinate()
     return nearestNeighbor(x,graph.x,1)


# Plot connectivity
# for i in range(1,len(graph.x)):
#      if 0 in graph.descendants(i): 
#           graph.plotNode(i,'r+',markeredgewidth=3)
#           path = astar(graph,i,0)
#           prev = i
#           for cur in path[1:]:
#                graph.plotEdge(prev,cur,'r',True,linewidth=1)
#                prev = cur

