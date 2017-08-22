from pinocchio.utils import *
import pinocchio as se3
from time import sleep
from numpy.linalg import inv,norm
import math
import time
import random
from pendulum import Pendulum
import matplotlib.pylab as plt
from specpath import acadoPath,dataRootPath,acadoTxtPath

RANDOM_SEED = 999 # int((time.time()%10)*1000)
print "Seed = %d" %  RANDOM_SEED
np .random.seed     (RANDOM_SEED)
random.seed         (RANDOM_SEED)

#env                 = Pendulum(2,withDisplay=True)       # Continuous pendulum
env                 = Pendulum(2,length=.5,mass=3.0,armature=.2,withDisplay=True)
env.withSinCos      = False             # State is dim-3: (cosq,sinq,qdot) ...
NX                  = env.nobs          # ... training converges with q,qdot with 2x more neurones.
NU                  = env.nu            # Control is dim-1: joint torque

env.vmax            = 100.
env.Kf              = np.diagflat([ 0.2, 2. ])
env.modulo          = False

env.DT              = 0.15
env.NDT             = 1
#env.umax            = 15.
#env.umax            = (15.,15.)
env.umax            = np.matrix([5.,10.]).T
NSTEPS              = 32

env.qlow[1] = -np.pi
env.qup [1] = np.pi


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
          colors = [ colorcycle.next() for _ in range(10000) ] \
              if len(self.existingConnex)>1 else ['k','.65']*10000
          

          for ix in range(len(self.x)): self.plotNode(ix)
          for (i0,i1),path in self.localPath.items():
               c = 2*self.connex[i0] + int(i0<i1)
               if withPath: plt.plot(path[:,0],path[:,1],colors[c])
               else:        plt.plot(path[[0,-1],0],path[[0,-1],1],colors[c])

     def play(self,idx0,idx1=None):
          env.display(graph.x[idx0])
          if idx1 is not None:
               time.sleep(.5)
               for x in self.localPath[idx0,idx1]: 
                    env.display(x)
                    time.sleep(.1)

     def save(self,path):
          import os
          try: os.mkdir(path)
          except: pass
          if path[-1]!='/': path+='/'
          for s in [ 'x', 'children', 'connex', 'localPath', 'localControl', 'edgeCost' ]:
               np.save(path+s+'.npy',self.__dict__[s])
     def load(self,path):
          if path[-1]!='/': path+='/'
          for s in [ 'x', 'children', 'connex', 'localPath', 'localControl', 'edgeCost' ]:
               self.__dict__[s] = np.load(path+s+'.npy')[()]
          self.nconnex = max(self.connex)+1
          self.existingConnex = sorted(list(set(self.connex)))
          self.x = [x for x in self.x]
          self.connex = [c for c in self.connex]

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
     def __init__(self,weights,nq=2):
          self.weights = weights
          self.nq = nq
     def __call__(self,x1,x2):
          dq = x1[:self.nq]-x2[:self.nq]
          dq = (dq+np.pi) % (2*np.pi) - np.pi
          dv = x1[self.nq:]-x2[self.nq:]
          sumsq = lambda x: np.sum(np.square(x))
          return np.sqrt( sumsq(dq)*self.weights[0] + sumsq(dv)*self.weights[1] )
          # cs1 = np.array(reduce(lambda x,y:x+y,[ (np.cos(x),np.sin(x)) for x in x1[:self.nq].flat]))
          # cs2 = np.array(reduce(lambda x,y:x+y,[ (np.cos(x),np.sin(x)) for x in x2[:self.nq].flat]))
          # return np.sqrt(np.sum(self.weights[0]*np.square(cs1-cs2)) 
          #                + np.sum(self.weights[1]*np.square(x1[self.nq:]-x2[self.nq:])) )

class NearestNeighbor:
     def __init__(self,hdistance = lambda x1,x2: norm(x1-x2)):
          self.hdistance = hdistance
     def __call__(self,x1, xs, nneighbor = 1, hdistance = None,fullSort=False):
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
          if fullSort:
               return np.argsort([ hdistance(x1,x2) for x2 in xs])[:nneighbor]
          else:
               return np.argpartition([ hdistance(x1,x2) for x2 in xs],
                                      nneighbor)[:nneighbor]

#wnorm = WeightedNorm([1,.4])
#wnorm = SO3Norm([1,1,.4])
wnorm = SO3Norm([1,.1])
nearestNeighbor = NearestNeighbor(wnorm)

from acado_connect import AcadoConnect

class ConnectAcado:
     def __init__(self):
          acado = self.acado = AcadoConnect(acadoPath,
                                            model=env.model,datadir=acadoTxtPath)
          acado.setTimeInterval(1.0)
          acado.options['steps']    = 25
          acado.options['shift']    = 0
          acado.options['iter']     = 20
          acado.options['friction'] = \
              "{0:f} {1:f}".format([env.Kf,]*2) if isinstance(env.Kf,float) \
              else  "{0:f} {1:f}".format(*env.Kf.diagonal())
          acado.options['umax']     = "%.2f %.2f" % tuple([x for x in env.umax])
          acado.options['armature'] = env.armature
          acado.setTimeInterval(1.5)

          self.threshold = 1e-3
          self.acado.setup_async()
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
     def opttime(self):
          return self.acado.opttime(self.idxBest)
     def times(self):
          return self.acado.times(self.idxBest)

connect = ConnectAcado()

def simplePrm(NSAMPLES = 1000, NCONNECT = 20, NBEST = 3, interactivePlot = False, nodeQueue = []):
     '''
     Basic PRM algorithm: sample <nsamples> points and try to connect each to <nconnect>
     neighbor. Minor variation: keep only the best <nbest> connections.
     nconnect: Number of nearest neighbors to try to connect with.
     '''
     connect.acado.debug(False) # just in case
     for _ in range(NSAMPLES):
          print "PRM sample #",len(graph.x)
          x = env.reset() if len(nodeQueue)==0 else nodeQueue.pop() 
          if not check(x): continue
          idx = graph.addNode(newConnex=True)           # Index of the new node
          graph.x[idx] = x                              # Add a new node for configuration x
          edges = []
          doConnect = False
          MODULO = np.matrix([-np.sign(x[0,0])*2*np.pi,0]).T
          for idx2 in nearestNeighbor(x,graph.x[:-1],NCONNECT):
               # From x to graph
               if connect(x,graph.x[idx2]): # or connect(x+MODULO,graph.x[idx2]):
                    edges.append( [ idx, idx2, connect.opttime(), 
                                    connect.states(),
                                    connect.controls() ] )
                    doConnect = True
               # From graph to x
               if connect(graph.x[idx2],x): # or connect(graph.x[idx2],x+MODULO):
                    edges.append( [ idx2, idx, connect.opttime(), 
                                    connect.states(),
                                    connect.controls() ] )

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

          
class GraphPend2(Graph):
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
          path = self.localPath[idx1,idx2]
          if withTruePath: 
               plt.plot(path[:,0],path[:,1],plotstr,**kwargs)
          else: 
               plt.plot(path[[0,-1],0],path[[0,-1],1],plotstr,**kwargs)


from capture_cursor import *

def nodeFromCursor():
     x = cursorCoordinate()
     return nearestNeighbor(x,graph.x,1, lambda x1,x2: norm(x1-x2[:2]))


from astar import astar
def pathFrom(idx,idx2=0):

     if idx2 not in graph.descendants(idx):
          #print idx2," not in descendance of idx:",graph.descendants(idx)
          raise Exception("%d not in descendance of %d:"%(idx2,idx))
          #graph.descendants(idx)

     traj = astar(graph,idx,idx2)
     prev = traj[0]
     path = []
     control = []
     times = []
     time = 0
     x = graph.x[idx]

     # q1 = q2 - mod2pi(q1,q2)
     mod2Pi = lambda x1,x2: np.vstack([np.round((x2[:2]-x1[:2])/2/np.pi)*2*np.pi,zero(2)])

     x0 = graph.x[idx]

     for cur in traj[1:]:
          T = graph.edgeCost[prev,cur]
          N = graph.localPath[prev,cur].shape[0]-1
          local = graph.localPath[prev,cur].copy()

          local -= mod2Pi(x0,local[:1,:].T).T
          x0 = local[-1:,:].T

          #print prev,cur,q,q0
          #assert( (q0%np.pi)-(q%np.pi)<1e-2 )
          #mod = round((q0-q)/2/np.pi)
          #q0 -= mod*2*np.pi
          
          #local[:,0] -= mod*2*np.pi
          #local -= mod2Pi(x,local[0,:])

          path   .append( local[:-1,:] )
          control.append( graph.localControl[prev,cur][:-1])
          times  .append( np.arange(0.,N)/N * T + time)
          time += T

          prev = cur
          #x = local[-1:,:]

     path.append( (graph.x[idx2]-mod2Pi(x0,graph.x[idx2])).T )
     control.append(zero(2).T)
     times.append(time)

     times = np.hstack(times)
     X = np.vstack(path)
     U = np.vstack(control)
     return X,U,times

def optpathFrom(idx,idx2=0):
     X,U,times = pathFrom(idx,idx2)
     time = times[-1]

     np.savetxt(acado.options['istate'],   np.vstack([times/time,X.T]).T )
     np.savetxt(acado.options['icontrol'], np.vstack([times/time,U.T]).T )
     acado.options['horizon'] = time
     acado.options['Tmax'] = 4*time

     #x0,x1 = graph.x[idx],graph.x[0]
     x0,x1 = X[0,:].T,X[-1,:].T
     #acado.debug()
     acado.iter = 100
     acado.options['steps'] = 50

     acado.run( x0,x1,autoInit=False)

     Xac = acado.states()
     Uac = acado.controls()

     return Xac,Uac,acado.times()

def test(idx = None):
     if idx is None: idx = random.randint(0,len(graph.x)-1)
     print "Opt Path with idx = ",idx
     X,_,_ = pathFrom(idx)
     Xac,_,_ = optpathFrom(idx)
     plt.plot(X[:,0],X[:,1],'r', linewidth=2)
     plt.plot(Xac[:,0],Xac[:,1],'g', linewidth=2)
     return X,Xac

# def optpolicy_mono(x0 = None, nbpoint = 1, nbcorrect = 1, withPlot = False):
#      x0 = env.reset() if x0 is None else x0.copy()
#      NQ = 2
#      PI = np.pi
#      PPI = 2*PI

#      solutions = []
#      t0 = time.time()
#      for idx in nearestNeighbor(x0,graph.x,nbpoint):
#           print "Try from ",idx,time.time()-t0
#           xnear = graph.x[idx]
#           dq = ((x0-xnear)[:NQ]+PI)%PPI - PI
#           x0[:NQ] = xnear[:NQ]+dq

#           X,U,times = pathFrom(idx)
#           ttime = times[-1]

#           np.savetxt(acado.options['istate'],   np.vstack([times/ttime,X.T]).T )
#           np.savetxt(acado.options['icontrol'], np.vstack([times/ttime,U.T]).T )
#           acado.options['horizon'] = ttime
#           acado.options['Tmax'] = 40*ttime

#           x1 = X[-1,:].T

#           acado.iter = 80
#           acado.options['steps'] = 20
#           acado.debug(False)
#           # acado.debug(True)
#           # acado.iter=5
#           try:
#                acado.run( x0,x1,autoInit=False)
#           except:
#                # print idx
#                # if idx not in [51,]:return
#                continue

#           Xac = acado.states()
#           Uac = acado.controls()

#           if withPlot:
#                plt.plot(X[:,0],X[:,1],'g', linewidth=2)
#                plt.plot(Xac[:,0],Xac[:,1],'r', linewidth=2)
#                plt.draw()

#           if norm(Xac[0,:]-x0.T)<1e-3 and norm(Xac[-1,:]-x1.T)<1e-3:
#                solutions.append([ acado.opttime(),Xac,Uac,acado.times() ])
#                print '\t\t\tSuccess'
#           else:
#                print 'Error when generating optimal trajectory: boundary constraints not respected'

#      if len(solutions)<nbcorrect:
#           raise Exception("Not enough points")

#      solutions = sorted(solutions,key=lambda s:s[0])
#      mintraj = solutions[0][0]
#      checktraj = solutions[nbcorrect-1][0]

#      if checktraj>1.1*mintraj:
#           raise Exception("Check point is too large")
     
#      #print [s[0] for s in solutions]
#      return solutions[0]


def optpolicy(x0 = None, nbpoint = 1, nbcorrect = 1, withPlot = False):
     x0 = env.reset() if x0 is None else x0.copy()
     NQ = 2
     PI = np.pi
     PPI = 2*PI

     solutions = []
     if 'horizon' in acado.options: del acado.options['horizon']
     if 'Tmax'    in acado.options: del acado.options['Tmax']
     acado.iter = 80
     acado.options['steps'] = 20
     acado.debug(False)
     # acado.debug(True)
     # acado.iter=5

     jobs = {}
     for idx in nearestNeighbor(x0,graph.x,nbpoint*2):
          #print "Try from ",idx,time.time()-t0

          xnear = graph.x[idx]
          dq = ((x0-xnear)[:NQ]+PI)%PPI - PI
          x0[:NQ] = xnear[:NQ]+dq

          try:               X,U,times = pathFrom(idx)
          except:            continue
          ttime = times[-1]

          jobid = acado.book_async()
          np.savetxt(acado.options['istate']+acado.async_ext(jobid),   np.vstack([times/ttime,X.T]).T )
          np.savetxt(acado.options['icontrol']+acado.async_ext(jobid), np.vstack([times/ttime,U.T]).T )

          x1 = X[-1,:].T

          acado.run_async( x0,x1,autoInit=False,jobid=jobid,
                           additionalOptions = ' --horizon=%.10f --Tmax=%.10f' % (ttime,5*ttime) )
          jobs[jobid] = [x0,x1]

          if withPlot:
               plt.plot(X[:,0],X[:,1],'g', linewidth=2)
               plt.draw()

          if len(jobs)==nbpoint: break

     for jobid,[x0,x1] in jobs.items():
          #print "Join ",jobid,time.time()-t0
          if acado.join(jobid,x0,x1): 
               Xac = acado.states(jobid)
               Uac = acado.controls(jobid)
               Tac = acado.times(jobid)
               cost = acado.opttime(jobid)
               solutions.append([ cost,Xac,Uac,Tac ])

               if withPlot:
                    plt.plot(Xac[:,0],Xac[:,1],'r', linewidth=2)
                    plt.draw()

     if len(solutions)<nbcorrect:
          raise Exception("Not enough points")

     solutions = sorted(solutions,key=lambda s:s[0])
     mintraj = solutions[0][0]
     checktraj = solutions[nbcorrect-1][0]

     if checktraj>1.1*mintraj:
          raise Exception("Check point is too large")
     
     #print [s[0] for s in solutions]
     return solutions[0]


def connectToZero(graph,idx0=1):
     for i in range(idx0,len(graph.x)):
          print 'Connect with ',i
          if 0 not in graph.children[i] and connect(graph.x[i],graph.x[0]):
               print 'yes'
               graph.addEdge(i,0,+1,localPath = connect.states(),localControl = connect.controls(),
                             cost = connect.opttime())
          if i not in graph.children[0] and connect(graph.x[0],graph.x[i]):
               print 'yes'
               graph.addEdge(0,i,+1,localPath = connect.states(),localControl = connect.controls(),
                             cost = connect.opttime())

def dataFromCursor():
     x0 = np.vstack([cursorCoordinate(),zero(2)])
     idx = nearestNeighbor(x0,[d.x0 for d in data])[0]
     return idx

def playFromCursor():
     x0 = np.vstack([cursorCoordinate(),zero(2)])
     idx = nearestNeighbor(x0,[d.x0 for d in data])[0]
     for x in data[idx].X: env.display(x); time.sleep(.1)
     return data[idx].x0
     


from collections import namedtuple
Data = namedtuple('Data', [ 'x0', 'X', 'cost', 'U', 'T' ])

def gridPolicy(step=.1):
     data = []
     trial = 0
     #shuffle = lambda l: random.sample(l,len(l))
     shuffle = lambda l:l
     for i2 in shuffle(np.arange(-np.pi+.01,np.pi-.01,step)):
          for i1 in shuffle(np.arange(-np.pi,np.pi,step)):
               trial += 1
               print 'Traj #',trial
               try:
                    x0 = np.matrix([i1,i2,0,0]).T
                    cost,X,U,T = optpolicy(x0,nbpoint=10,nbcorrect=1,withPlot=False)
                    data.append( Data(x0=x0,X=X,cost=cost,U=U,T=T) )
               except:
                    print 'Failure at #',trial
                    data.append( Data(x0=x0,X=[],cost=100000.,U=zero(2).T,T=[]) )

     return data

# def refineGrid(data,NTRIAL=1000,NNEIGHBOR=8,RANDQUEUE=[],PERCENTAGE=.95):
#      for trial in range(NTRIAL):

#           idx0 = RANDQUEUE.pop() if len(RANDQUEUE)>0 else random.randint(0,len(data)-1)
#           d0 = data[idx0]
#           x0 = d0.x0
#           NQ=2
#           PI = np.pi
#           PPI = 2*PI
#           #print "Trial #",trial,idx0,x0[:2].T
          
#           for idx2 in nearestNeighbor(x0, [ d.x0 for d in data ],NNEIGHBOR,fullSort=True ):
#                if idx2 == idx0: continue

#                d2 = data[idx2]
#                if d2.cost>d0.cost*PERCENTAGE: continue

#                ttime,X,U,T = d2.cost,d2.X,d2.U,d2.T

#                xnear = X[:1,:].T
#                dq = ((x0-xnear)[:NQ]+PI)%PPI - PI
#                x0mod = x0.copy()
#                x0mod[:NQ] = xnear[:NQ]+dq

#                x1 = X[-1,:].T

#                np.savetxt(acado.options['istate'],   np.vstack([T/ttime,X.T]).T )
#                np.savetxt(acado.options['icontrol'], np.vstack([T/ttime,U.T]).T )

#                acado.options['horizon'] = ttime
#                acado.options['Tmax'] = 2*ttime

#                #print '\t\t\tTry ',x0[:2].T,'(%.4f) by '%d0.cost,xnear[:2].T,'(%.4f)'%d2.cost

#                acado.iter = 80
#                acado.options['steps'] = 20
#                try:
#                     acado.run(x0mod,x1,autoInit=False)
#                     #print '\t\t\t\t\twas %.5f now %.5f'%(d0.cost,acado.opttime())
#                except: 
#                     #print '\t\t\t\t\tfailed...'
#                     continue
               
#                X = acado.states()
#                if norm(X[0,:]-x0mod.T)>1e-3 or norm(X[-1,:]-x1.T)>1e-3: 
#                     print 'Error in refine grid: boundary constraints not respected'
#                     continue

#                #print 'From %4d:%2.5f ... from %4d(%2.5f):%2.5f' %( idx0,d0.cost,idx2,d2.cost,acado.opttime())
#                if acado.opttime()<d0.cost:
#                     data[idx0] = Data( x0=x0, X=X, U=acado.controls(), 
#                                        T=acado.times(), cost=acado.opttime() )
#                     print "#%4d: %4d is best from %4d" %(trial,idx0,idx2),"\t(%.3f vs %.3f)"%(acado.opttime(),d0.cost)
#                     break

def refineGrid(data,NTRIAL=1000,NNEIGHBOR=8,RANDQUEUE=[],PERCENTAGE=.95):
     NQ = env.model.nq;   PI = np.pi;    PPI = 2*PI
     if 'horizon' in acado.options: del acado.options['horizon']
     if 'Tmax'    in acado.options: del acado.options['Tmax']
     acado.iter = 80
     acado.options['steps'] = 20
     for trial in range(NTRIAL):

          idx0 = RANDQUEUE.pop() if len(RANDQUEUE)>0 else random.randint(0,len(data)-1)
          d0 = data[idx0]
          x0 = d0.x0
          #print "Trial #",trial,idx0,x0[:2].T
          
          jobs = {}
          for idx2 in nearestNeighbor(x0, [ d.x0 for d in data ],NNEIGHBOR+1,fullSort=True ):
               if idx2 == idx0: continue

               d2 = data[idx2]
               if d2.cost>d0.cost*PERCENTAGE: continue

               jobid = acado.book_async()
               ttime,X,U,T = d2.cost,d2.X,d2.U,d2.T

               xnear = X[:1,:].T
               dq = ((x0-xnear)[:NQ]+PI)%PPI - PI
               x0mod = x0.copy()
               x0mod[:NQ] = xnear[:NQ]+dq

               x1 = X[-1,:].T

               np.savetxt(acado.options['istate']  +acado.async_ext(jobid), np.vstack([T/ttime,X.T]).T )
               np.savetxt(acado.options['icontrol']+acado.async_ext(jobid), np.vstack([T/ttime,U.T]).T )

               acado.run_async(x0mod,x1,autoInit=False,jobid=jobid,
                               additionalOptions= ' --horizon=%.10f --Tmax=%.10f' % (ttime,2*ttime))
               jobs[jobid] = [x0mod,x1]
     
          for jobid,[x0mod,x1] in jobs.items():
               if acado.join(jobid,x0,x1):
                    if acado.opttime(jobid)<d0.cost:
                         data[idx0] = Data( x0  = x0, 
                                            X   = acado.states  (jobid),
                                            U   = acado.controls(jobid), 
                                            T   = acado.times   (jobid),
                                            cost= acado.opttime (jobid) )
                    print "#%4d: %4d is best from %4d" %(trial,idx0,idx2),"\t(%.3f vs %.3f)"%(acado.opttime(jobid),d0.cost)
                    break


def densifyPrm(graph,NTRIAL=1000,PAUSEFREQ=50):
     # trial = 0
     # for idx1,x1 in enumerate(graph.x):
     #      for idx2,x2 in enumerate(graph.x):
     for trial in xrange(NTRIAL):
               if not trial % PAUSEFREQ: 
                    print 'Time for a little break ... 2s',time.ctime()
                    time.sleep(1)
               print 'trial #',trial
               idx1=random.randint(0,len(graph.x)-1)
               idx2=random.randint(0,len(graph.x)-1)
               if idx1==idx2: continue
               if idx2 in graph.children[idx1]: continue
               try:
                    X,U,T = optpathFrom(idx1,idx2)
               except:
                    continue
               print 'Connect %d to %d'%(idx1,idx2)
               graph.addEdge(idx1,idx2,+1,
                             localPath=X,
                             localControl=U,
                             cost=acado.opttime())


def connexifyPrm(graph,NTRIAL = 1000,PAUSEFREQ = 50,NCONNECT=5):
     for trial in xrange(NTRIAL):
          if not trial % PAUSEFREQ: 
               print 'Time for a little break ... 2s',time.ctime()
               time.sleep(1)
          print 'trial #',trial
          while True:  # Take two sampled not connected
               idx1=random.randint(0,len(graph.x)-1)
               idx2=random.randint(0,len(graph.x)-1)
               if idx1 == idx2: continue
               des = graph.descendants(idx1)
               if idx2 not in des: break
          for ides in random.sample(des,NCONNECT) if len(des)>NCONNECT else des:
               if idx2 in graph.children[ides]: continue
               if connect(graph.x[ides],graph.x[idx2]):
                    graph.addEdge(ides,idx2,+1,
                                  localPath=connect.states(),
                                  localControl=connect.controls(),
                                  cost=connect.opttime())
                    print 'Connect %d to %d'%(ides,idx2)
                         
               

# --- MAIN -----------------------------------------------------------------------
# --- MAIN -----------------------------------------------------------------------
# --- MAIN -----------------------------------------------------------------------

plt.ion()

graph = GraphPend2()
graph.addNode(newConnex=True)
graph.x[0] = zero(4)

connect.acado.setTimeInterval(1.)
acado=connect.acado
acado.options['printlevel']=1

graph.load(dataRootPath)
'''
for i in range(5):
     simplePrm(20,10,10,True)
     print 'Sleeping 10 ... it is time for a little CTRL-C ',time.ctime()
     time.sleep(10)

graph.save(dataRootPath+'_100pts')

### Filling the prm with additional points at low speed.
env.vup[:] = 3.
env.vlow[:] = -3.
for i in range(5):
     simplePrm(10,50,50,False)
     print 'Sleeping 10 ... it is time for a little CTRL-C ',time.ctime()
     time.sleep(10)

graph.save(dataRootPath+'_200pts')

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
'''

### Filling the prm with additional points close to joint limit.
env.qlow = np.matrix([-5, -np.pi])
env.qup  = np.matrix([ 5, -.6*np.pi])
env.vlow[:] = -.5
env.vup[:] = .5
prevSize = len(graph.x)
for i in range(5):
     simplePrm(10,20,20,False)
     print 'Sleeping 10 ... it is time for a little CTRL-C ',time.ctime()
     time.sleep(10)

env.qlow = np.matrix([-5,  .6*np.pi])
env.qup  = np.matrix([ 5,     np.pi])
env.vlow[:] = -.5
env.vup[:] = .5
prevSize = len(graph.x)
for i in range(5):
     simplePrm(10,20,20,False)
     print 'Sleeping 10 ... it is time for a little CTRL-C ',time.ctime()
     time.sleep(10)

'''
print 'Connect all points to zero (at least tries)',time.ctime()
connectToZero(graph)
print 'Densify PRM',time.ctime()
'''

densifyPrm(graph)
connexifyPrm(graph)

graph.save(dataRootPath)

### Generate the grid ##########################################################

RANDOM_SEED =  int((time.time()%10)*1000)
print "Seed = %d" %  RANDOM_SEED
np .random.seed     (RANDOM_SEED)
random.seed         (RANDOM_SEED)

'''
dataflat = np.load(dataRootPath+'/grid.npy')
data=[]
for i,d in enumerate(dataflat): data.append(Data(*d))
'''
print 'Generate the grid',time.ctime()
data = gridPolicy()
np.save(dataRootPath+'/grid_first.npy',data)
print 'Refine the grid',time.ctime()
refineGrid(data,5000,NNEIGHBOR=30,PERCENTAGE=.9)
np.save(dataRootPath+'/grid.npy',data)

'''
D = np.vstack([ np.hstack([d.x0.T,d.U[:1,:],np.matrix(d.cost)]) for d in data])

plt.subplot(2,2,1)
plt.scatter(D[:,0].flat,D[:,1].flat,c=D[:,-1].flat,s=70,alpha=.8,linewidths=0,vmax=5.)
plt.subplot(2,2,3)
plt.scatter(D[:,0].flat,D[:,1].flat,c=D[:,4].flat,s=70,alpha=.8,linewidths=0)
plt.subplot(2,2,4)
plt.scatter(D[:,0].flat,D[:,1].flat,c=D[:,5].flat,s=70,alpha=.8,linewidths=0)
plt.colorbar()
'''
