from pinocchio.utils import *
from numpy.linalg import inv, norm
import math
import time
import random
import matplotlib.pylab as plt

from prm import *
from specpath import acadoBinDir, acadoTxtPath
from quadcopterpendulum import QuadcopterPendulum
from acado_connect import AcadoConnect

dataRootPath = 'data/planner/quadcopterpendulum'


# --- BICOPTER -------------------------------------------------------------------
from quadcopter_steering import GraphQuadcopter as GraphQuadcopterPendulum

def config(acado, label, env=None):
    acado.options['maxAngle'] = np.pi / 2
    acado.options['maxAnglePend'] = np.pi / 4
    acado.options['printlevel'] = 1
    # acado.options['g'] = env.g
    if env is not None:
        acado.setDims(env.nq, env.nv)
    if 'shift' in acado.options: del acado.options['shift']
    if env is not None:
        acado.options['umax'] = "%.2f %.2f %.2f %.2f" % tuple([x for x in env.umax])

    if label == "connect":
        # if 'icontrol' in acado.options: del acado.options['icontrol']
        acado.debug(False)
        acado.iter = 50
        acado.options['steps'] = 20
        acado.options['acadoKKT'] = 0.0001
        acado.options['icontrol'] = acadoTxtPath + 'guess.clt'
        acado.setTimeInterval(5.)

    elif label == "traj":
        if 'horizon' in acado.options: del acado.options['horizon']
        if 'Tmax' in acado.options: del acado.options['Tmax']
        acado.debug(False)
        acado.iter = 80
        acado.options['steps'] = 20
        acado.options['icontrol'] = acadoTxtPath + 'guess.clt'
        acado.options['acadoKKT'] = 0.0001

    elif label == "policy":
        if 'horizon' in acado.options: del acado.options['horizon']
        if 'Tmax' in acado.options: del acado.options['Tmax']
        acado.debug(False)
        acado.iter = 80
        acado.options['steps'] = 20
        acado.options['icontrol'] = acadoTxtPath + 'guess.clt'
        acado.options['acadoKKT'] = 0.0001

    elif label == "refine":
        if 'horizon' in acado.options: del acado.options['horizon']
        if 'Tmax' in acado.options: del acado.options['Tmax']
        acado.debug(False)
        acado.iter = 80
        acado.options['steps'] = 20
        acado.options['icontrol'] = acadoTxtPath + 'guess.clt'
        acado.options['acadoKKT'] = 0.0001

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


class QuadcopterPendulumStateDiff:
    '''Given two states (of the bicopter) returns the delta (Lie exponential) 
    to go from x1 to x2.'''

    def __call__(self, x1, x2):
        return x2 - x1


# ---
class AcadoQuadConnect(AcadoConnect):
    def __init__(self,*args,**kwargs):
        AcadoConnect.__init__(self,*args,**kwargs)
        #self.setup_async()
    def states(self,jobid=None):
        X = AcadoConnect.states(self,jobid)
        return X[:,[0,1,2,3,4,6,7,8,9,10]]
    def buildInitGuess(self,x0,x1,jobid=None):
        X,U,T = self.guess(x0,x1,self.options)
        if 'istate' in self.options:
            X13 = zero([X.shape[0],self.NQ+self.NV+3])
            X13[:,:self.NQ] = X[:,:self.NQ]
            X13[:,self.NQ+1:self.NQ+1+self.NV] = X[:,self.NQ:self.NQ+self.NV]
            np.savetxt(self.stateFile('i',jobid),np.vstack([T/T[-1], X.T]).T)
        if U is not None and 'icontrol' in self.options:
            np.savetxt(self.controlFile('i',jobid),np.vstack([T/T[-1], U.T]).T)
        return X,U,T
    # def run(self,x0,x1,*args,**kwargs):
    #     jobid = self.run_async(x0,x1,*args,**kwargs)
    #     try:
    #         return self.join(jobid,x0,x1,timeout = 10)
    #     except:
    #         return False


env = QuadcopterPendulum(withDisplay=False)

env.sphericalObstacle = True
env.obstacleSize = 1.
env.obstaclePosition = np.matrix([2., 2., 0.]).T

#acado = AcadoQuadConnect(acadoBinDir + "connect_quadcopter",
acado = AcadoConnect(acadoBinDir + "connect_quadcopter_pendulum",
                     datadir=acadoTxtPath)
config(acado, 'connect', env)
acado.setDims(env.nq,env.nv)


#env.qup = np.matrix([2., 2., 2., np.pi/2, np.pi/2, np.pi/5, np.pi/5 ]).T
env.qup = np.matrix([1., 1., 1., np.pi/8, np.pi/8, np.pi/10, np.pi/10 ]).T
env.vup = np.matrix([.2]*5 + [.01]*2).T

env.qmax = np.matrix([5., 5., 5., np.pi/2, np.pi/2, np.pi/3, np.pi/3]).T
env.vmax = np.matrix([5]*7).T
env.xmax = np.concatenate([env.qmax,env.vmax])

env.qlow = -env.qup
env.vlow = -env.vup
env.xmin = -env.xmax


