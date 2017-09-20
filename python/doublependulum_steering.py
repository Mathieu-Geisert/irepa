from pinocchio.utils import *
from numpy.linalg import inv, norm
import math
import time
import random
import matplotlib.pylab as plt

from prm import *
from specpath import acadoBinDir, acadoTxtPath
from pendulum import Pendulum
from acado_connect import AcadoConnect

dataRootPath = 'data/planner/doublependulum'


# --- BICOPTER -------------------------------------------------------------------


class GraphDoublePendulum(Graph):
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


def config(acado, label, env=None):
    acado.options['printlevel'] = 1
    # acado.options['g'] = env.g
    if env is not None:
        pattern = " ".join(["{%d:f}"%i for i in range(env.nv)])
        acado.options['friction'] = \
            pattern.format(*[env.Kf,]*env.nv) if isinstance(env.Kf,float) \
            else  pattern.format(*env.Kf.diagonal())
        acado.setDims(env.nq, env.nv)
        acado.options['armature'] = env.armature
        acado.options['umax']     = "%.2f %.2f" % tuple([x for x in env.umax])
    if 'shift' in acado.options: del acado.options['shift']

    if label == "connect":
        # if 'icontrol' in acado.options: del acado.options['icontrol']
        acado.debug(False)
        acado.iter = 50
        acado.options['steps'] = 25
        acado.options['acadoKKT'] = 0.0001
        acado.options['icontrol'] = acadoTxtPath + 'guess.clt'
        acado.options['istate'] = acadoTxtPath + 'guess.stx'
        acado.setTimeInterval(2.)

    elif label == "traj":
        if 'horizon' in acado.options: del acado.options['horizon']
        if 'Tmax' in acado.options: del acado.options['Tmax']
        acado.debug(False)
        acado.iter = 80
        acado.options['steps'] = 20
        acado.options['icontrol'] = acadoTxtPath + 'guess.clt'
        acado.options['istate'] = acadoTxtPath + 'guess.stx'
        acado.options['acadoKKT'] = 0.0001

'''
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
'''

class ConnectAcado(ConnectAbstract):
    def __init__(self, acado):
        self.acado = acado

    def __call__(self, x1, x2, verbose=False):
        try:
            return self.acado.run(x1, x2)
        except:
            return False

    def states(self):
        return self.acado.states()

    def controls(self):
        return self.acado.controls()

    def cost(self):
        return self.acado.opttime()

    def times(self):
        return self.acado.times()

    def time(self):
        return self.acado.opttime()


class DoublePendulumStateDiff:
     def __init__(self,NQ):
          self.NQ = NQ

     def __call__(self,x1,x2):
          '''return dx modulo 2*PI such that x2 = x1 + dx, dx minimal'''
          PI = np.pi;     PPI = 2*PI
          dx = x2-x1
          dq = (dx[:self.NQ]+PI)%PPI - PI
          dv =  dx[self.NQ:]
          return np.concatenate([dq,dv])



env = Pendulum(2,length=.5,mass=3.0,armature=.2,withDisplay=False)
env.withSinCos      = False             # State is dim-3: (cosq,sinq,qdot) ...
env.vmax            = 100.
env.Kf              = np.diagflat([ 0.2, 2. ])
env.modulo          = False
env.DT              = 0.15
env.NDT             = 1
env.umax            = np.matrix([5.,10.]).T

env.qlow[1]         = -np.pi
env.qup             = np.matrix([ 2*np.pi, np.pi ]).T
env.qlow            = -env.qup
env.vup             = np.matrix([ 3, ]*2).T
env.vlow            = -env.vup

env.xmax            = np.matrix([ 3*np.pi, np.pi, 8, 8]).T
env.xmin            = -env.xmax


acado = AcadoConnect(acadoBinDir + "connect_double_pendulum",
                     datadir=acadoTxtPath)
config(acado, 'connect', env)
acado.setDims(env.nq,env.nv)

