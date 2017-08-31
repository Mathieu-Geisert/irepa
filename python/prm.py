from pinocchio.utils import *
import time
from numpy.linalg import inv,norm
import math
import random
import matplotlib.pylab as plt
from collections import namedtuple

class Graph:
     def __init__(self):
          self.x        = []
          self.children = {}
          self.connex   = []    # ID of the connex component the node is belonging to
          self.nconnex = 0      # number of connex components
          self.existingConnex = [] # List of existing connex component ID
          self.states = {}
          self.controls = {}
          self.edgeCost = {}
          self.edgeTime = {}

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
                 states = None, controls = None, cost = -1, time = 0. ):
          '''
          Add edge from first to second. Also add edge from second to first if orientation
          is null.
          '''
          assert( first in self.children and second in self.children )

          self.children[first].append(second)
          self.states[ first,second ]     = states
          self.controls[ first,second ]   = controls
          self.edgeCost[ first,second ]   = cost
          self.edgeTime[first,second]     = time

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
          del self.states[idx,idx2]
          del self.controls[idx,idx2]
          del self.edgeCost[idx,idx2]
          del self.edgeTime[idx,idx2]

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

     # --- GRAPH DISPLAY ---
     def plotNode(self,idx,plotstr='k+',**kwargs):
          print 'Not implemented yet. To be implemented in inheriting classes'
     def plotEdge(self,idx1,idx2,plotstr='k',withTruePath = False,**kwargs):
          print 'Not implemented yet. To be implemented in inheriting classes'
     def plot(self,withTruePath=False):
          for ix in range(len(self.x)): self.plotNode(ix)
          for (i0,i1),path in self.states.items():
               self.plotEdge(i0,i1,withTruePath=withTruePath)
     def play(self,idx0,idx1=None):
          print 'Not implemented yet. To be implemented in inheriting classes'

     # --- GRAPH IO ---
     def save(self,path):
          import os
          try: os.mkdir(path)
          except: pass
          if path[-1]!='/': path+='/'
          for s in [ 'x', 'children', 'connex', 'states', 'controls', 'edgeCost', 'edgeTime' ]:
               np.save(path+s+'.npy',self.__dict__[s])

     def load(self,path):
          if path[-1]!='/': path+='/'
          for s in [ 'x', 'children', 'connex', 'states', 'controls', 'edgeCost', 'edgeTime' ]:
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

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
class ConfigCheckerAbstract:
     def __call__(self,x): return True

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
class DistanceAbstract:
     def __call__(self,x1,x2): return np.norm(x1-x2)

class DistanceWeightedNorm:
     def __init__(self,weights):
          self.weights = weights
     def __call__(self,x1,x2):
          return np.sqrt(np.sum(self.weights*np.square(x1-x2)))

class DistanceSO3:
     '''
     Assume that the state is composed of NQ SO(3) elements th1...th_NQ followed 
     by a tail of Euclidean elements. 
     The SO(3) elements are measured modulo 2*PI. All components are weighted.
     '''
     def __init__(self,weights,nq=2):
          self.weights = weights
          self.nq = nq
     def __call__(self,x1,x2):
          dq = x1[:self.nq]-x2[:self.nq]
          dq = (dq+np.pi) % (2*np.pi) - np.pi
          dv = x1[self.nq:]-x2[self.nq:]
          sumsq = lambda x: np.sum(np.square(x))
          return np.sqrt( sumsq(dq)*self.weights[0] + sumsq(dv)*self.weights[1] )

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

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
ConnectResults = namedtuple('ConnectResults', [ 'cost', 'states', 'controls', 'time' ])
class ConnectAbstract:
     def __call__(self,x1,x2,verbose=False):
          return True
     def cost(self):
          '''Return the cost of previous call'''
          return 0.
     def states(self):
          '''Return the state traj of previous call'''
          return []
     def controls(self):
          '''Return the control traj of previous call'''
          return []
     def time(self):
          '''Return the trajectory total time of previous call'''
          return 0.
     def results(self):
          return ConnectResults(cost=self.cost(), states=self.states(), 
                                controls=self.controls(), time=self.time() )

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
class PRM:
  def __init__(self,graph,sampler,checker,nearestNeighbor,connect):
       self.graph   = graph
       self.sampler = sampler
       self.checker = checker
       self.nearestNeighbor = nearestNeighbor
       self.connect = connect
  def __call__(self, NSAMPLES = -1, NCONNECT = 20, NBEST = 3, VERBOSE = False, RANDQUEUE = []):
     '''
     Basic PRM algorithm: sample <nsamples> points and try to connect each to <nconnect>
     neighbor. Minor variation: keep only the best <nbest> connections.
     nconnect: Number of nearest neighbors to try to connect with.
     '''
     graph   = self.graph
     sampler = self.sampler
     checker = self.checker
     nearest = self.nearestNeighbor
     connect = self.connect

     if NSAMPLES < 0: NSAMPLES = len(RANDQUEUE)

     for _ in range(NSAMPLES):
          x = sampler() if len(RANDQUEUE)==0 else RANDQUEUE.pop() 
          if not checker(x): continue

          idx = graph.addNode(newConnex=True)           # Index of the new node
          graph.x[idx] = x                              # Add a new node for configuration x
          edges = []

          for idx2 in nearest(x,graph.x[:-1],NCONNECT):
               # From x to graph
               if connect(x,graph.x[idx2]):
                    edges.append( [ idx, idx2, connect.results() ] )
               # From graph to x
               if connect(graph.x[idx2],x):
                    edges.append( [ idx2, idx, connect.results() ] )

          edges = sorted(edges, key = lambda e: e[2])[:min(len(edges),NBEST)]
          for idx1, idx2, edge in edges:
               graph.addEdge(idx1,idx2,+1,**edge._asdict())
               graph.renameConnex(graph.connex[idx2],graph.connex[idx1])

          if VERBOSE:
               print "PRM sample #",len(graph.x)-1
               graph.plotNode(idx)
               for idx1,idx2,_ in edges: 
                    print '\t ...Connect %d to %d'%(idx1,idx2)
                    graph.plotEdge(idx1,idx2,withTruePath=True,
                                   color= 'k' if idx1<idx2 else '.5')
               plt.draw()


  def connectToZero(self,idx0=1,VERBOSE=False):
     '''Tries to connect all nodes of the graph to node <idx0>.'''   
     graph = self.graph
     connect = self.connect
     for i in range(idx0,len(graph.x)):
          if VERBOSE: print 'Connect with ',i
          if 0 not in graph.children[i] and connect(graph.x[i],graph.x[0]):
               if VERBOSE: print '\tTo <%d>: yes' % idx0
               graph.addEdge(i,0,+1,**connect.results()._asdict())
          if i not in graph.children[0] and connect(graph.x[0],graph.x[i]):
               if VERBOSE: print '\tFrom <%d>: yes' % idx0
               graph.addEdge(0,i,+1,**connect.results()._asdict())



  def connexifyPrm(self,NTRIAL = 1000,PAUSEFREQ = 50,NCONNECT=5,VERBOSE=False):
     '''Try to create additionnal edges between nodes of the graph that are not connected.'''  
     graph = self.graph
     connect = self.connect
     for trial in xrange(NTRIAL):
          if not trial % PAUSEFREQ: 
               print 'Time for a little break ... 1s',time.ctime()
               time.sleep(1)
          if VERBOSE: print 'trial #',trial
          while True:  # Take two samples that are not connected
               idx1=random.randint(0,len(graph.x)-1)
               idx2=random.randint(0,len(graph.x)-1)
               if idx1 == idx2: continue
               des = graph.descendants(idx1)
               if idx2 not in des: break
          for ides in random.sample(des,NCONNECT) if len(des)>NCONNECT else des:
               assert(idx2 not in graph.children[ides])
               if connect(graph.x[ides],graph.x[idx2]):
                    graph.addEdge(ides,idx2,+1,**connect.results()._asdict())
                    if VERBOSE: print 'Connect %d to %d'%(ides,idx2)
