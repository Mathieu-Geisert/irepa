'''
Initialize a double pendulum, run acado from scratch to get a control
trajectory, then rollout this trajectory on the system.
The test might be used to check that the dynamics inside acado and inside python are the same.
'''

from pendulum import Pendulum
from scipy.optimize import *
from pinocchio.utils import *
import pinocchio as se3
import numpy as np
from numpy import sin,cos
from numpy.linalg import norm
import time
import signal
import matplotlib.pyplot as plt
import acado_runner
plt.ion()

env = Pendulum(2,length=.5,mass=3.0,armature=10.)

acado = acado_runner.AcadoRunner("/home/nmansard/src/pinocchio/pycado/build/unittest/discrete_double_pendulum")

env.umax        = 15.
env.vmax        = 100
env.modulo      = False
NSTEPS          = 50
env.DT          = 0.2 #25e-3
env.NDT         = 1
env.Kf          = 1.0
x0              = np.matrix([0.3,0.3,0.,0.]).T

def controlPD(withPlot = True):
    '''
    Compute a control trajectory by rollout a PD controller.
    '''
    Kp=50.; Kv=2*np.sqrt(Kp)*.5
    ratioK = 1.0
    #xdes = np.matrix([3.14,0,0,0]).T
    xdes = np.matrix([0.,0,0,0]).T
    K = np.matrix(np.hstack([np.diagflat([ratioK*Kp,Kp]),np.diagflat([np.sqrt(ratioK)*Kv,Kv])]))
    print (-K*(env.x-xdes)).T

    hx = []
    hu = []
    for i in range(int(10/env.DT)):
        if abs(i*env.DT%1)<env.DT/2: print "time=",env.DT*i
        u = np.clip( -K*(env.x-xdes), -env.umax,env.umax)
        hx.append(env.x.copy())
        hu.append(u.copy())
        env.step(u)
        env.render()
    X=np.hstack(hx).T
    U=np.hstack(hu).T
    if withPlot:
        plt.subplot(2,1,1)
        plt.plot(X)
        plt.subplot(2,1,2)
        plt.plot(U)
    return X,U


env.reset(x0)
env.render()

acado.options['horizon'] = NSTEPS*env.DT
acado.options['steps']   = NSTEPS
acado.options['friction'] = env.Kf
#acado.options['decay'] = 10.

print "Compute initial trajectory ... "
acado.initrun(env.x[:2],env.x[2:])
print "                           ... ok"

from acado_runner import f2a
U=np.vstack(f2a(acado.controlFile))
X=np.vstack(f2a(acado.stateFile))

def rollout(x0,U):
    env.reset(x0)
    hx = [ env.x.copy().T, ]
    for u in U:
        env.step(u[1:])
        env.render()
        hx.append( env.x.copy().T )
    return np.vstack(hx)

X2 = rollout(x0,U)
  
#plt.plot(X[:,1:-1])
#plt.plot(X2)
