#
# MPC second order
#

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
plt.ion()

acadoexe = "/home/nmansard/src/pinocchio/pycado/build/unittest/pendulum2o"

def readAcadoControl(filename = "/tmp/control.txt"):
    us = []
    N = 0
    ls = open(filename).readlines()
    _,a,b,c   = [ float(s) for s in ls[0].split() ]
    t,_,_,_   = [ float(s) for s in ls[1].split() ]
    return a,b,c,t

f2a = lambda filename: [ [float(s) for s in l.split() ] for l in open(filename).readlines()]
fstate = "/tmp/states.txt"
fcontrol = "/tmp/control.txt"
rs = lambda : np.array(f2a(fstate))[:,1:2]
#rp = lambda : (np.array(f2a(fstate))[:,1] + np.pi) % (2*np.pi) - np.pi
rp = lambda : np.array(f2a(fstate))[:,1]
rc = lambda : np.array(f2a(fcontrol))[:,1:]


env = Pendulum(1)
x0 = np.matrix([-3.0,0.0]).T

env.NDT = 10
env.umax = 100.
env.vmax = 100.
env.Kf = 0.

env.x = x0.copy()

# Initial optim
cmd = acadoexe + " --steps=10 -T 5 -N 100 -p %.10f -v %.10f " % ( env.x[0,0], env.x[1,0] )
os.system(cmd)

#plt.plot(rp())
#plt.axis([0,50,-3.14,3.14])
#plt.axis([0,50,-10,10])

xprev = env.x.copy()
for i in range(150):
    cmd = acadoexe + " --steps=10 -T 5 -N 5 -p %.20f -v %.20f -t 1 -i /tmp/control.txt -j /tmp/states.txt" % ( env.x[0,0], env.x[1,0] )
    os.system(cmd)
    a,b,c,DT = readAcadoControl()

    env.DT = DT = DT/10
    xprev = env.x.copy()
    for loop in range(10):
        t = loop*DT
        u = a + b*t + .5*c*t*t
        #print "U%d = %.3f" % (i,u)
        env.step([u,])
        env.render()

    # plt.plot(range(i,21+i),rp())
    # plt.show()
    # time.sleep(.1)
    # raw_input("Press Enter to continue...")
