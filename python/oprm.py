from collections import namedtuple
from prm import PRM
from astar import astar
import numpy as np
from pinocchio.utils import *
import random

class OptimalPRM(PRM):
  PathFromResults = namedtuple("PathFromResults",['states','controls','times','cost'])

  @staticmethod
  def makeFromPRM(prm,acado,stateDiff = None):
       kwarg=prm.__dict__
       return OptimalPRM(acado=acado,stateDiff=stateDiff,**kwarg)

  def __init__(self,graph,sampler,checker,nearestNeighbor,connect,
               acado,stateDiff = None, **kwargs ):
       '''
       Init from a PRM <graph> and an optimizer <acado>.
       <configDiff> is a method than returns the smallest step dx to go from x1 to x2: x2 = x1+dx.
       It is used in optPolicy method.
       '''
       if stateDiff is None: stateDiff = lambda x1,x2: x2-x1
       PRM.__init__(self,graph,sampler,checker,nearestNeighbor,connect)
       self.acado       = acado
       self.stateDiff   = stateDiff

  def pathFrom(self,idx,idx2=0):
     '''
     Return a full path from node <idx> to node <idx2>, by stacking the partial paths 
     extracted by astar from the graph.
     If the partial paths are not continuous one to the following, the method forces the continuity.
     '''
     graph = self.graph
     acado = self.acado

     if idx2 not in graph.descendants(idx):
          raise Exception("%d not in descendance of %d:"%(idx2,idx))

     traj = astar(graph,idx,idx2)

     states   = []
     controls = []
     times    = []
     cost     = 0.
     time     = 0.
     prev     = traj[0]
     xprev    = graph.x[idx]

     for cur in traj[1:]:
          T = graph.edgeTime[prev,cur]
          N = graph.states[prev,cur].shape[0]-1

          X = graph.states[prev,cur]
          dx = xprev - X[:1,:].T

          states  .append( graph.states  [prev,cur][:-1,:] + dx.T)
          controls.append( graph.controls[prev,cur][:-1,:])
          times   .append( np.arange(0.,N)/N * T + time )
          time += T
          cost += graph.edgeCost[prev,cur]
          xprev = X[-1:,:].T + dx
          prev  = cur

     states  .append( xprev.T )
     controls.append( zero(2).T )
     times   .append( time )

     return self.PathFromResults(states=np.vstack(states),controls=np.vstack(controls),
                                 cost=cost,times=np.hstack(times))

  def optPathFrom(self,idx,idx2=0):
     '''
     Get a path from the graph, them optimize it and return the result.
     '''
     acado = self.acado
     traj  = self.pathFrom(idx,idx2)

     ttime = traj.times[-1]
     np.savetxt(acado.options['istate'],   np.vstack([traj.times/ttime,traj.states  .T]).T )
     np.savetxt(acado.options['icontrol'], np.vstack([traj.times/ttime,traj.controls.T]).T )

     acado.run( traj.states[0,:].T,traj.states[-1,:].T,autoInit=False,
                additionalOptions = ' --horizon=%.10f' % (ttime))

     if acado.opttime()>ttime:
       raise  RuntimeError("Optimized path is longer than initial guess: from %d to %d" % (idx,idx2) )

     return self.PathFromResults(states=acado.states(),controls=acado.controls(),
                                 cost=acado.cost(),times=acado.times())


  def optPolicy(self,x0, nbpoint = 1, nbcorrect = 1, withPlot = False):
     '''
     Compute an optimal policy from one arbitrary point x0 (maybe not in the PRM) to 0.
     '''
     x0 = x0.copy()
     acado = self.acado
     graph = self.graph
     nearest = self.nearestNeighbor

     jobs = {}
     for idx in nearest(x0,graph.x,nbpoint*2,fullSort=True):
          if idx==0: continue
          xnear = graph.x[idx]
          x0    = xnear - self.stateDiff(x0,xnear)

          try:               traj = self.pathFrom(idx)
          except:            continue
          ttime = traj.times[-1]

          jobid = acado.book_async()
          np.savetxt(acado.stateFile  ('i',jobid),  np.vstack([traj.times/ttime,traj.states  .T]).T )
          np.savetxt(acado.controlFile('i',jobid),  np.vstack([traj.times/ttime,traj.controls.T]).T )

          acado.run_async( x0,traj.states[-1,:].T,
                           autoInit=False,jobid=jobid,
                           additionalOptions = ' --horizon=%.10f' % (ttime) )
          jobs[jobid] = [ x0,traj.states[-1,:].T]

          if withPlot:
               plt.plot(traj.states[:,0],traj.states[:,1],'g', linewidth=.5)
               plt.draw()

          if len(jobs)==nbpoint: break

     solutions = []
     for jobid,[x0,x1] in jobs.items():
          if acado.join(jobid,x0,x1): 
               Xac  = acado.states  (jobid)
               Uac  = acado.controls(jobid)
               Tac  = acado.times   (jobid)
               cost = acado.opttime (jobid)
               solutions.append(self.PathFromResults(cost=cost,states=Xac,controls=Uac,times=Tac))

               if withPlot:
                    plt.plot(Xac[:,0],Xac[:,1],'r', linewidth=2)
                    plt.draw()

     if len(solutions)<nbcorrect:
          raise Exception("Not enough points")

     solutions = sorted(solutions,key=lambda s:s.cost)
     if solutions[0].cost>1.1*solutions[nbcorrect-1].cost:
          raise Exception("Check point is too large")

     return solutions[0]

  def densifyPrm(self,NTRIAL=1000,PAUSEFREQ=0,CHECKCHILD=False,VERBOSE=False):
       '''
       Try to directly connect some nodes in the PRM.
       If CHECKCHILD is True, then the solver do not tries to find shorter paths for 
       nodes that are already directly connected.
       '''  
       graph = self.graph
       for trial in xrange(NTRIAL):
               if PAUSEFREQ>0 and not trial % PAUSEFREQ: 
                    print 'Time for a little break ... 2s',time.ctime()
                    time.sleep(1)
               idx1=random.randint(0,len(graph.x)-1)
               idx2=random.randint(0,len(graph.x)-1)
               if idx1==idx2: continue
               if CHECKCHILD and idx2 in graph.children[idx1]: continue
               if VERBOSE>1: print 'trial #%d: %3d to %3d' %( trial,idx1,idx2 )
               try:
                    traj = self.optPathFrom(idx1,idx2)
               except Exception as exc:
                    print exc
                    continue
               if VERBOSE>1: print '\t\tConnect %d to %d'%(idx1,idx2)
               ttime = traj.times[-1]
               if VERBOSE: 
                 prevtime = self.pathFrom(idx1,idx2).times[-1]
                 if prevtime < .9*ttime:
                   print '\t\tWas %.2f -- Now %.2f' % (prevtime,ttime)
               graph.addEdge(idx1,idx2,+1,time=ttime,**traj._asdict())
