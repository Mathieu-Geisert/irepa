from pinocchio.utils import *
from numpy.linalg import inv,norm
import math
import time
import random
import matplotlib.pylab as plt

from prm import *
from specpath import acadoBinDir,acadoTxtPath
from bicopter import Bicopter
from acado_connect import AcadoConnect

dataRootPath = 'data/planner/bicopter'
     
# --- BICOPTER -------------------------------------------------------------------


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
     acado.options['thetamax']    = np.pi/2
     acado.options['printlevel']    = 1
     if env is not None:
         acado.setDims(env.nq,env.nv)
     if 'shift' in acado.options: del acado.options['shift']
     if env is not None:
          acado.options['umax']     = "%.2f %.2f" % tuple([x for x in env.umax])
          
     if label == "connect":
          if 'icontrol' in acado.options: del acado.options['icontrol']
          acado.debug(False)
          acado.iter                = 20
          acado.options['steps']    = 20
          acado.setTimeInterval(5.)

     elif label == "traj":
          if 'horizon' in acado.options: del acado.options['horizon']
          if 'Tmax'    in acado.options: del acado.options['Tmax']
          acado.debug(False)
          acado.iter                = 80
          acado.options['steps']    = 20
          acado.options['icontrol'] = acadoTxtPath+'guess.clt'
          
     elif label == "policy":
          if 'horizon' in acado.options: del acado.options['horizon']
          if 'Tmax'    in acado.options: del acado.options['Tmax']
          acado.debug(False)
          acado.iter                = 80
          acado.options['steps']    = 20
          acado.options['icontrol'] = acadoTxtPath+'guess.clt'

     elif label == "refine":
          if 'horizon' in acado.options: del acado.options['horizon']
          if 'Tmax'    in acado.options: del acado.options['Tmax']
          acado.debug(False)
          acado.iter                = 80
          acado.options['steps']    = 20
          acado.options['icontrol'] = acadoTxtPath+'guess.clt'

class ConnectAcado(ConnectAbstract):
     def __init__(self,acado):
          self.acado = acado

     def __call__(self,x1,x2,verbose=False):
          try: return self.acado.run(x1,x2)
          except: return False

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

class BicopterStateDiff:
     '''Given two states (of the bicopter) returns the delta (Lie exponential) 
     to go from x1 to x2.'''
     def __call__(self,x1,x2):
          return x2-x1


# ---

env     = Bicopter(withDisplay=False)
acado   = AcadoConnect(acadoBinDir+"connect_bicopter",
                       datadir=acadoTxtPath)
config(acado,'connect',env)
