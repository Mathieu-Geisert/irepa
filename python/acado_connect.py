from pinocchio.utils import *
import pinocchio as se3
from time import sleep
from numpy.linalg import inv,norm
import math
import time
import heapq
import random
import matplotlib.pylab as plt
from acado_runner import *
from pendulum import Pendulum

RANDOM_SEED = int((time.time()%10)*1000)
print "Seed = %d" %  RANDOM_SEED
np .random.seed     (RANDOM_SEED)
random.seed         (RANDOM_SEED)

env                 = Pendulum(1,withDisplay=False)       # Continuous pendulum
env.withSinCos      = False             # State is dim-3: (cosq,sinq,qdot) ...
NX                  = env.nobs          # ... training converges with q,qdot with 2x more neurones.
NU                  = env.nu            # Control is dim-1: joint torque

env.vmax            = 100.
env.Kf              = 0.2
env.modulo          = False

env.DT              = 0.05
env.NDT             = 1
T                   = .5
NSTEPS              = int(round(T/env.DT))

DT = env.DT

#x1 = rand(2)
x1 = np.matrix([ .2, .1]).T
#x2 = rand(2)
#x2 = np.matrix([ 0., .0]).T 
x2 = np.matrix([ .2, -.1]).T 

### Compute polynomial trajectory from x1 to x2
N = 3                   # Polynom degree
A = zero([4,N+1])
b = zero(4)

x = x1.copy()
hx = []
hu = []
for i in range(NSTEPS):
     t = i*DT
     A[0,:] =  [ t**i for i in range(N+1) ]
     A[1,1:] = [ i*t**(i-1) for i in range(1,N+1) ]
     A[2,:] =  [ T**i for i in range(N+1) ]
     A[3,1:] = [ i*T**(i-1) for i in range(1,N+1) ]

     b[:2] = x
     b[2:] = x2

     a = [ (i-1)*i*t**(i-2) for i in range(2,N+1) ]*(inv(A)*b)[2:]

     tau = se3.rnea(env.model,env.data,x[:1],x[1:],a)
     
     hx.append(x.copy())
     hu.append(tau)

     x[1] += a*DT
     x[0] += x[1]*DT

hx.append(x2)
hu.append(zero(1))
X = np.hstack(hx).T
U = np.hstack(hu).T

plt.ion()
plt.plot(np.arange(NSTEPS+1)*DT,X)
plt.plot(np.arange(NSTEPS+1)*DT,U)


from acado_runner import AcadoRunner

class AcadoConnect(AcadoRunner):
     def __init__(self,path="/home/nmansard/src/pinocchio/pycado/build/unittest/connect_pendulum"):
          AcadoRunner.__init__(self,path)
     def setTimeInterval(self,T,T0=0.001):
          acado.options['horizon'] = T
          acado.options['Tmin'] = T0
          acado.options['Tmax'] = T*2

     def buildInitGuess(self,x0,x1):
          T      = self.options['horizon']
          NSTEPS = self.options['steps']
          DT     = T / NSTEPS

          N = 3                   # Polynom degree
          A = zero([4,N+1])       # Matrix of t**i coefficients
          b = zero(4)             # Vector of x,xdot references

          hx = []
          hu = []
          x = x1.copy()
          for i in range(NSTEPS):
               t = i*DT
               A[0,:] =  [ t**i for i in range(N+1) ]
               A[1,1:] = [ i*t**(i-1) for i in range(1,N+1) ]
               A[2,:] =  [ T**i for i in range(N+1) ]
               A[3,1:] = [ i*T**(i-1) for i in range(1,N+1) ]

               b[:2] = x
               b[2:] = x2

               a = [ (i-1)*i*t**(i-2) for i in range(2,N+1) ]*(inv(A)*b)[2:]    # Acceleration
               tau = se3.rnea(env.model,env.data,x[:1],x[1:],a)                 # Torque
     
               hx.append(x.copy())
               hu.append(tau)

               x[1] += a*DT                                                     # Integation v
               x[0] += x[1]*DT                                                  # Integration p

          hx.append(x2)            # Append terminal state
          hu.append(zero(1))       # Append terminal control (meaningless)
          X = np.hstack(hx).T
          U = np.hstack(hu).T
          
          guessX = np.vstack([ np.arange(NSTEPS+1)/float(NSTEPS),
                               X.T, zero(NSTEPS+1).T ]).T
          guessU = np.vstack([ np.arange(NSTEPS+1)/float(NSTEPS),
                               U.T ]).T

          np.savetxt(self.options['istate'],guessX)
          np.savetxt(self.options['icontrol'],guessU)

          return X,U

     def run(self,x0,x1,autoInit=True):
          self.options['finalpos'] = ' '.join([ '%.20f'%f for f in x1.flat[:1] ])
          self.options['finalvel'] = ' '.join([ '%.20f'%f for f in x1.flat[1:] ])
          if autoInit and 'istate' in self.options and 'icontrol' in self.options:
               self.buildInitGuess(x0,x1)
          return AcadoRunner.run(self,x0.flat[:1],x0.flat[1:])

     def initrun(self,x0,x1,i=100,autoInit = True):
          self.options['finalpos'] = ' '.join([ '%.20f'%f for f in x1.flat[:1] ])
          self.options['finalvel'] = ' '.join([ '%.20f'%f for f in x1.flat[1:] ])
          if autoInit and 'istate' in self.options and 'icontrol' in self.options:
               self.buildInitGuess(x0,x1)
          return AcadoRunner.initrun(self,x0.flat[:1],x0.flat[1:])

#acado = AcadoRunner("/home/nmansard/src/pinocchio/pycado/build/unittest/connect_pendulum")
acado = AcadoConnect()
# acado.options['horizon'] = T
# acado.options['Tmin'] = 0.001
# acado.options['Tmax'] = T*2
acado.setTimeInterval(T)
acado.options['steps']    = NSTEPS
acado.options['shift']    = 0
acado.options['iter']     = 100
acado.options['friction'] = 0.2
#del acado.options['istate']
# acado.options['finalpos'] = x2[0,0]
# acado.options['finalvel'] = x2[1,0]

# while horizon T is optimized, timescale should be rescaled between 0 and 1 
# see http://acado.sourceforge.net/doc/html/d4/d29/example_002.html
# guessX = np.vstack([ np.arange(NSTEPS+1)/float(NSTEPS),
#                      X.T, zero(NSTEPS+1).T ]).T
# guessU = np.vstack([ np.arange(NSTEPS+1)/float(NSTEPS),
#                      U.T ]).T

# np.savetxt('/tmp/state.txt',guessX)
# np.savetxt('/tmp/control.txt',guessU)

acado.options['istate']='/tmp/guess.stx'
acado.options['icontrol'] = '/tmp/guess.ctl'
    
acado.debug()
acado.options['iter']=100
#u,cost = acado.run(X[0,:1],X[0,1:])
u,cost = acado.run(X[0,:].T,X[-1,:].T)
    
