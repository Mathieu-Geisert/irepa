from pendulum import Pendulum
from scipy.optimize import *
from pinocchio.utils import *
import pinocchio as se3
from numpy import sin,cos
from numpy.linalg import norm
import time
import signal
import matplotlib.pyplot as plt
import os
import copy
plt.ion()

env = Pendulum(1)

env.DT  = .25
env.NDT =  10
# env.umax = 100.
# env.vmax = 100.
env.Kf = 0.

HORIZON = 5.0
NHORIZON = int(HORIZON/env.DT)

f2a = lambda filename: [ [float(s) for s in l.split() ] for l in open(filename).readlines()]

class AcadoRunner:
    def __init__(self,path= "/home/nmansard/src/pinocchio/pycado/build/unittest/pendulum"):
        self.exe = path
        self.dataroot = '/tmp/mpc'
        self.datactl  = '.ctl'
        self.datastx  = '.stx'
        self.controlFile  = self.dataroot + self.datactl
        self.stateFile    = self.dataroot + self.datastx
        self.options = { 
            'horizon'   : HORIZON,
            'steps'     : NHORIZON,
            'iter'      : 3,
            'icontrol'  : self.controlFile,
            'istate'    : self.stateFile,
            'ocontrol'  : self.controlFile,
            'ostate'    : self.stateFile,
            'shift'     : 1,
            }
    def run(self,pos,vel,opts = None):
        if not opts: opts = self.options
        self.cmd = self.exe + ' ' \
            + ' '.join([ '--%s=%s' % (k,str(v)) for k,v in opts.items() ]) \
            + ' -p %.20f -v %.20f > /dev/null' % ( pos,vel )
        os.system(self.cmd)
        ctls = f2a(self.controlFile)
        stts = f2a(self.stateFile)
        return ctls[0][1:],stts[-1][-1]
    def initrun(self,pos,vel,i=100):
        opt = copy.copy(self.options)
        del opt['icontrol']
        del opt['istate']
        del opt['shift']
        opt['iter'] = i
        return self.run(pos,vel,opt)
        
    def states(self):
        return np.array(f2a(self.stateFile))[:,1:3]
    def controls(self):
        return np.array(f2a(self.controlFile))[:,1:2]
    def costs(self):
        return np.array(f2a(self.stateFile))[:,3]

acado = AcadoRunner()

database = []

with_plot = 0
with_render = 0

for rollout in range(10000):
    env.reset()#(np.matrix([2.3,0.]).T)
    #env.render()
    if not (rollout%10): 
        print 'Computing the initial trajectory <%d> from '%rollout,env.x.T
        time.sleep(2)
    acado.initrun(env.x[0,0],env.x[1,0])
    #print acado.cmd
    #print 'Trajectory computed'

    # Initial optim
    for controlcycle in range(5):
        u,cost = acado.run(env.x[0,0],env.x[1,0])
        
        database.append( [ env.x[0,0], env.x[1,0], u[0], cost ] )
        env.step(u)
        if with_render: 
            env.render()
            print "U%d = %.3f\t\t%.1f" % (i,u[0],cost)
        if norm(env.x)<1e-4: break
        if with_plot:
            plt.plot(range(controlcycle,controlcycle+21),acado.states()[:,0])
            plt.plot([controlcycle,controlcycle],[-10,10])
            raw_input("Press Enter to continue...")
            plt.figure(1).axes[0].lines.pop()

D=np.array(database)
np.save(open('database.np','w'),D)

def pl3(database):
    from mpl_toolkits.mplot3d import Axes3D
    import pylab
    D = np.array(database)
    fig = pylab.figure()
    ax = Axes3D(fig)
    ax.scatter(D[:,0],D[:,1],D[:,2])
    #plt.show()

