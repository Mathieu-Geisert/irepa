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
          self.setDims()

     def setDims(self):
          if self.model is None: self.NQ,self.NV = 1,1; return
          self.NQ = self.model.nq
          self.NV = self.model.nv

     def setTimeInterval(self,T,T0=0.001):
          self.options['horizon'] = T
          self.options['Tmin'] = T0
          self.options['Tmax'] = T*4

     def buildInitGuess(self,x0,x1):
          T      = self.options['horizon']
          NSTEPS = self.options['steps']
          DT     = T / NSTEPS

          withControl = self.model is not None and self.data is not None

          N = 3                   # Polynom degree
          C = zero([4,N+1])       # Matrix of t**i coefficients
          b = zero(4)             # Vector of x,xdot references
 
          t = 0.0
          C[0,:] =  [ t**i for i in range(N+1) ]
          C[1,1:] = [ i*t**(i-1) for i in range(1,N+1) ]
          C[2,:] =  [ T**i for i in range(N+1) ]
          C[3,1:] = [ i*T**(i-1) for i in range(1,N+1) ]
          
          assert(self.NQ == self.NV)
          P = []
          V = []
          A = []
          U = []
          for iq in range(self.NQ):
          
               b[:2] = x0[iq::self.NQ]
               b[2:] = x1[iq::self.NQ]
         
               c = inv(C)*b

               P.append( np.vstack([ [t**i for i in range(N+1)]*c for t in np.arange(0,T+DT/2,DT) ]) )
               V.append (np.vstack([ [i*t**(i-1) for i in range(1,N+1)]*c[1:] for t in np.arange(0,T+DT/2,DT) ]))
               A.append( np.vstack([ [i*(i-1)*t**(i-2) for i in range(2,N+1)]*c[2:] for t in np.arange(0,T+DT/2,DT) ]))
               #X = np.hstack([P,V])

          X = np.hstack(P+V)

          if 'armature' in self.options:
               armature = self.options['armature']
               dyninv = lambda p,v,a: se3.rnea(self.model,self.data,p,v,a)+armature*a
          else:
               dyninv = lambda p,v,a: se3.rnea(self.model,self.data,p,v,a)

          U = np.vstack([ dyninv(p.T,v.T,a.T).T for p,v,a in zip(np.hstack(P),np.hstack(V),np.hstack(A)) ]) if self.model is not None else []
              

          # while horizon T is optimized, timescale should be rescaled between 0 and 1 
          # see http://acado.sourceforge.net/doc/html/d4/d29/example_002.html
          guessX = np.vstack([ np.arange(NSTEPS+1)/float(NSTEPS), X.T, zero(NSTEPS+1).T ]).T
          np.savetxt(self.options['istate'],guessX)

          if withControl:
               guessU = np.vstack([ np.arange(NSTEPS+1)/float(NSTEPS),   U.T ]).T
               np.savetxt(self.options['icontrol'],guessU)

          return X,U

     def run(self,x0,x1,autoInit=True):
          self.options['finalpos'] = ' '.join([ '%.20f'%f for f in x1[:self.NQ].flat ])
          self.options['finalvel'] = ' '.join([ '%.20f'%f for f in x1[self.NQ:].flat ])
          if autoInit:               self.buildInitGuess(x0,x1)
          return AcadoRunner.run(self,x0[:self.NQ].flat,x0[self.NQ:].flat)

     def initrun(self,x0,x1,i=100,autoInit = True):
          self.options['finalpos'] = ' '.join([ '%.20f'%f for f in x1[:self.NQ].flat ])
          self.options['finalvel'] = ' '.join([ '%.20f'%f for f in x1[self.NQ:].flat ])
          if autoInit:               self.buildInitGuess(x0,x1)
          return AcadoRunner.run(self,x0[:self.NQ].flat,x0[self.NQ:].flat)

     def rerun(self): return AcadoRunner.run(self)

     def states(self):
         '''The problem is Mayer-based, hence the cost is not part of the state file.'''
         return np.array(f2a(self.stateFile))[:,1:]

     def params(self):
         return np.array(f2a(self.options['oparam']))[:,1:]
     def times(self):
          '''Return times of state and control samplings.'''
          N = self.options['steps']
          return np.arange(0.,N+1)/N * self.opttime()
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
    
