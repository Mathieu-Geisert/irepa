from pinocchio.utils import *
from numpy.linalg import inv, norm
import math
import time
import random
import matplotlib.pylab as plt

from prm import *
from specpath import acadoBinDir, acadoTxtPath
from quadcopter import Quadcopter
from acado_connect import AcadoConnect

dataRootPath = 'data/planner/quadcopter'


# --- BICOPTER -------------------------------------------------------------------


class GraphQuadcopter(Graph):
    '''Specialization of Graph to have the display dedicated to quadcopter.'''

    def __init__(self):
        Graph.__init__(self)

    def plotState(self, q, v=None, color='r', marker='o', lseg=.1):
        q = q.flat
        plt.plot([q[0]-0.1*np.cos(q[5]), q[0]+0.1*np.cos(q[5])],
                 [q[1]+0.1*np.sin(q[5]), q[1]-0.1*np.sin(q[5])],
                 linestyle='-', marker=marker, linewidth=1, color=color)
        # plt.plot([q[1]-0.5*np.cos(q[4]), q[1]+0.5*np.cos(q[4])],
        #          [q[2]+0.5*np.sin(q[4]), q[2]-0.5*np.sin(q[4])],
        #          linestyle='--', marker=marker, linewidth=1, color=color)

    def plotNode(self, idx, plotstr='k', **kwargs):
        x = self.x[idx]
        if x is None: return
        self.plotState(x, color=plotstr)
        plt.plot(x[0, 0], x[1, 0], plotstr, **kwargs)

    def plotEdge(self, idx1, idx2, plotstr='k', withTruePath=False, **kwargs):
        path = self.states[idx1, idx2]
        if withTruePath:
            plt.plot(path[:, 0], path[:, 1], plotstr, **kwargs)
        else:
            plt.plot(path[[0, -1], 0], path[[0, -1], 1], plotstr, **kwargs)


def config(acado, label, env=None):
    acado.options['maxAngle'] = np.pi / 2
    acado.options['printlevel'] = 1
    # acado.options['g'] = env.g
    if env is not None:
        acado.setDims(env.nq, env.nv)
    if 'shift' in acado.options: del acado.options['shift']
    if env is not None:
        acado.options['umax'] = "%.2f %.2f %.2f %.2f" % tuple([x for x in env.umax])

    if env.sphericalObstacle:
        acado.options['sphericalObstacle'] = "%.2f %.2f %.2f %.2f"\
                                             % tuple([env.obstacleSize,\
                                                      env.obstaclePosition[0], \
                                                      env.obstaclePosition[1], \
                                                      env.obstaclePosition[2]])
    if label == "connect":
        # if 'icontrol' in acado.options: del acado.options['icontrol']
        acado.debug(False)
        acado.iter = 30
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


class QuadcopterStateDiff:
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


env = Quadcopter(withDisplay=False)
#acado = AcadoQuadConnect(acadoBinDir + "connect_quadcopter",
acado = AcadoConnect(acadoBinDir + "connect_quadcopter",
                     datadir=acadoTxtPath)
config(acado, 'connect', env)
acado.setDims(env.nq,env.nv)

