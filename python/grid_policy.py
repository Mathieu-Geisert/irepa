from pinocchio.utils import *
from numpy.linalg import inv,norm
import math
import time
import random
import matplotlib.pylab as plt
from collections import namedtuple

from prm import PRM
from oprm import OptimalPRM


def reggrid(arrs):
    '''
    Return a NxM array with M=len(arrs+1) and N=len(arr)*len(grid(*arrs))
    On each row of the result is a vector regularly sampled in the intervals arr,*arrs.
    '''
    arr = arrs[0]; arrs = arrs[1:]
    if len(arrs)==0:         return np.expand_dims(arr.flat,1)
    G = reggrid(arrs)
    return np.hstack([ np.expand_dims(np.concatenate([arr.flat]*len(G)),1),
                       G.repeat(len(arr),0)])


class GridPolicy:

   Data = namedtuple('Data', [ 'x0', 'X', 'cost', 'U', 'T' ])

   def __init__(self,oprm):
        self.optimalPRM = oprm
        self.data = []
        self.shuffle = lambda l: random.sample(l,len(l))
        #self.shuffle = lambda l:l
        self.nbpoint = 16               # Number of point sampled from PRM when computing a traj
        self.costError = 1000000.

   def setGrid(self,lower,upper,step):
        if isinstance(lower,np.matrix): lower = lower.flat
        if isinstance(upper,np.matrix): upper = upper.flat
        if isinstance(step ,np.matrix): step  = step .flat
        if isinstance(step, float)    : step  = [step,] * len(upper)
        if lower is None              : lower = - upper

        self.grid = reggrid( [ np.arange(l,u,s) for l,u,s in zip(lower,upper,step) ] )
        return self.grid

   def reconfigAcado(self):
        '''Change the setup of acado to correspond to the need of refineGrid.'''

   def sample(self,subsample=1,verbose=False):
     '''Generate the point of the grid.'''
     data = self.data
     grid = self.grid 
     oprm = self.optimalPRM
     checkifexist = len(data) > 0   
     if verbose: print '### Sample grid: %d points to evaluate' % len(self.grid[::subsample,:])
     for trial,x0 in enumerate(self.shuffle(self.grid[::subsample,:])):
          if verbose: print 'Traj #',trial
          try:
               x0 = np.matrix(x0).T
               if checkifexist and len([ True for d in data if np.allclose(d.x0,x0) ])>0:
                    if verbose: print '\t...already done'
                    continue
               traj = oprm.optPolicy(x0,nbpoint=self.nbpoint,nbcorrect=1,withPlot=False)
               data.append( self.Data(x0=x0,X=traj.states,cost=traj.cost,U=traj.controls,T=traj.times) )
          except:
               print 'Failure at #',trial
               data.append( self.Data(x0=x0,X=[],cost=self.costError,U=zero(2).T,T=[]) )

   def refineGrid(self,NTRIAL=1000,NNEIGHBOR=8,RANDQUEUE=[],PERCENTAGE=.95,verbose=False):
     '''Refine the grid to smoothen the underlying functions (policy, value).'''
     data      = self.data
     acado     = self.optimalPRM.acado
     nearest   = self.optimalPRM.nearestNeighbor
     stateDiff = self.optimalPRM.stateDiff
     
     if NTRIAL<0: NTRIAL = len(RANDQUEUE)
     for trial in range(NTRIAL):
          idx0 = RANDQUEUE.pop() if len(RANDQUEUE)>0 else random.randint(0,len(data)-1)
          d0 = data[idx0]
          x0 = d0.x0
          if verbose: print "Trial #",trial,idx0,x0[:2].T
          
          jobs = {}
          for idx2 in nearest(x0, [ d.x0 for d in data ],NNEIGHBOR+1,fullSort=True ):
               d2 = data[idx2]
               if idx2 == idx0 or d2.cost>d0.cost*PERCENTAGE: continue

               jobid       = acado.book_async()
               ttime,X,U,T = d2.cost,d2.X,d2.U,d2.T
               x0          = X[:1,:].T - stateDiff(x0,X[:1,:].T)
               x1          = X[-1,:].T
               np.savetxt(acado.stateFile  ('i',jobid), np.vstack([T/ttime,X.T]).T )
               np.savetxt(acado.controlFile('i',jobid), np.vstack([T/ttime,U.T]).T )

               acado.run_async(x0,x1,autoInit=False,jobid=jobid,
                               additionalOptions= ' --horizon=%.10f --Tmax=%.10f' % (ttime,2*ttime))
               jobs[jobid] = [x0,x1,idx2]
     
          for jobid,[x0mod,x1,idx2] in jobs.items():
               if acado.join(jobid,x0,x1):
                    if acado.cost(jobid)<d0.cost:
                         data[idx0] = self.Data( x0  = x0, 
                                                 X   = acado.states  (jobid),
                                                 U   = acado.controls(jobid), 
                                                 T   = acado.times   (jobid),
                                                 cost= acado.cost    (jobid) )
                         if verbose: print "#%4d: %4d is best from %4d" %(trial,idx0,idx2),\
                                 "\t(%.3f vs %.3f)"%(acado.cost(jobid),d0.cost)
                         d0 = data[idx0]

   # --- IO DISP --------------------------------------------------------------------
   # --- IO DISP --------------------------------------------------------------------
   # --- IO DISP --------------------------------------------------------------------

   def load(self,filename):
        for d in np.load(filename): self.data.append(self.Data(*d))
        
   def save(self,filename):
        np.save(filename,self.data)

   def np(self):
       return np.vstack([ np.hstack([d.x0.T,d.U[:1,:],np.matrix(d.cost)]) for d in self.data])

   def plot(self,colorAxes,xaxis=0,yaxis=1,layout = None,**kwargs):
       D = self.np()
       if isinstance(colorAxes,int): colorAxes = [ colorAxes, ]
       plotargs = { 's':70,'alpha':.8,'linewidths':0,'vmax':5. }
       plotargs.update(kwargs)
       for i,c in enumerate(colorAxes):
           if layout is not None: plt.subplot(layout[0],layout[1],i)
           plt.scatter(D[:,xaxis].flat,D[:,yaxis].flat,c=D[:,c].flat,**plotargs)

