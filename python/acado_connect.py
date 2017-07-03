from pinocchio.utils import *
import pinocchio as se3
from time import sleep
from numpy.linalg import inv,norm
from acado_runner import *
from acado_runner import AcadoRunner

class AcadoConnect(AcadoRunner):
     def __init__(self,path="/home/nmansard/src/pinocchio/pycado/build/unittest/connect_pendulum",
                  model=None,data=None):
          AcadoRunner.__init__(self,path)
          self.model = model
          if self.model is not None: self.data = model.createData()
          self.options['istate']   ='/tmp/guess.stx'
          self.options['icontrol'] = '/tmp/guess.ctl'
          self.options['oparam'] = '/tmp/mpc.prm'

     def setTimeInterval(self,T,T0=0.001):
          self.options['horizon'] = T
          self.options['Tmin'] = T0
          self.options['Tmax'] = T*2

     def buildInitGuess(self,x0,x1):
          T      = self.options['horizon']
          NSTEPS = self.options['steps']
          DT     = T / NSTEPS

          withControl = self.model is not None and self.data is not None

          N = 3                   # Polynom degree
          A = zero([4,N+1])       # Matrix of t**i coefficients
          b = zero(4)             # Vector of x,xdot references

          hx = []
          hu = []
          x = x0.copy()
          for i in range(NSTEPS):
               t = i*DT
               A[0,:] =  [ t**i for i in range(N+1) ]
               A[1,1:] = [ i*t**(i-1) for i in range(1,N+1) ]
               A[2,:] =  [ T**i for i in range(N+1) ]
               A[3,1:] = [ i*T**(i-1) for i in range(1,N+1) ]

               b[:2] = x
               b[2:] = x1

               a = [ (i-1)*i*t**(i-2) for i in range(2,N+1) ]*(inv(A)*b)[2:]    # Acceleration
               if withControl:
                    tau = se3.rnea(self.model,self.data,x[:1],x[1:],a)          # Torque
                    hu.append(tau)
     
               hx.append(x.copy())

               x[1] += a*DT                                                     # Integation v
               x[0] += x[1]*DT                                                  # Integration p

          hx.append(x1)            # Append terminal state
          hu.append(zero(1))       # Append terminal control (meaningless)
          X = np.hstack(hx).T
          U = np.hstack(hu).T
          
          # while horizon T is optimized, timescale should be rescaled between 0 and 1 
          # see http://acado.sourceforge.net/doc/html/d4/d29/example_002.html
          guessX = np.vstack([ np.arange(NSTEPS+1)/float(NSTEPS), X.T, zero(NSTEPS+1).T ]).T
          np.savetxt(self.options['istate'],guessX)

          if withControl:
               guessU = np.vstack([ np.arange(NSTEPS+1)/float(NSTEPS),   U.T ]).T
               np.savetxt(self.options['icontrol'],guessU)

          return X,U

     def run(self,x0,x1,autoInit=True):
          self.options['finalpos'] = ' '.join([ '%.20f'%f for f in x1.flat[:1] ])
          self.options['finalvel'] = ' '.join([ '%.20f'%f for f in x1.flat[1:] ])
          if autoInit:               self.buildInitGuess(x0,x1)
          return AcadoRunner.run(self,x0.flat[:1],x0.flat[1:])

     def initrun(self,x0,x1,i=100,autoInit = True):
          self.options['finalpos'] = ' '.join([ '%.20f'%f for f in x1.flat[:1] ])
          self.options['finalvel'] = ' '.join([ '%.20f'%f for f in x1.flat[1:] ])
          if autoInit:               self.buildInitGuess(x0,x1)
          return AcadoRunner.initrun(self,x0.flat[:1],x0.flat[1:])

     def rerun(self): return AcadoRunner.run(self)

     def states(self):
         '''The problem is Mayer-based, hence the cost is not part of the state file.'''
         return np.array(f2a(self.stateFile))[:,1:]

     def params(self):
         return np.array(f2a(self.options['oparam']))[:,1:]
     
     def opttime(self):
          return self.params()[0,0]
     

if __name__ == '__main__':

     from pendulum import Pendulum

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

     x1 = np.matrix([ .2, .1]).T
     x2 = np.matrix([ .2, -.1]).T 

     acado = AcadoConnect(model=env.model)
     acado.setTimeInterval(T)
     acado.options['steps']    = NSTEPS
     acado.options['shift']    = 0
     acado.options['iter']     = 100
     acado.options['friction'] = 0.2
     
     # acado.debug()
     u,cost = acado.run(x1,x2)
    
