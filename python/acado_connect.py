from pinocchio.utils import *
import pinocchio as se3
from time import sleep
from numpy.linalg import inv,norm
from acado_runner import *
from acado_runner import AcadoRunner

class AcadoConnect(AcadoRunner):
     def __init__(self,path="/home/nmansard/src/pinocchio/pycado/build/unittest/connect_pendulum",
                  model=None,data=None,datadir='/tmp/'):
          AcadoRunner.__init__(self,path)
          self.model = model
          if self.model is not None: self.data = model.createData()
          self.options['istate']   = datadir+'guess.stx'
          self.options['icontrol'] = datadir+'guess.ctl'
          self.options['ocontrol'] = datadir+'mpc.ctl'
          self.options['ostate']   = datadir+'mpc.stx'
          self.options['oparam']   = datadir+'mpc.prm'
          self.setDims()

     def setDims(self):
          if self.model is None: self.NQ,self.NV = 1,1; return
          self.NQ = self.model.nq
          self.NV = self.model.nv

     def setTimeInterval(self,T,T0=0.001):
          self.options['horizon'] = T
          self.options['Tmin'] = T0
          self.options['Tmax'] = T*4

     def buildInitGuess(self,x0,x1,jobid=None):
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
          np.savetxt(self.options['istate']+self.async_ext(jobid),guessX)

          if withControl:
               guessU = np.vstack([ np.arange(NSTEPS+1)/float(NSTEPS),   U.T ]).T
               np.savetxt(self.options['icontrol']+self.async_ext(jobid),guessU)

          return X,U

     def run(self,x0,x1,autoInit=True):
          if autoInit:               self.buildInitGuess(x0,x1)
          AcadoRunner.run(self,states={'init':  [ x0[:self.NQ].flat,x0[self.NQ:].flat ],
                                       'final': [ x1[:self.NQ].flat,x1[self.NQ:].flat ] })
          return self.checkResult(True,x0,x1)

     def checkResult(self,ret,x0=None,x1=None,jobid=None,threshold=1e-3):
          if not ret: return False
          if x0 is not None or x1 is not None: X = self.states(jobid)
          if x0 is not None and norm(x0.T-X[ 0,:]) > threshold: return False
          if x1 is not None and norm(x1.T-X[-1,:]) > threshold: return False
          return True

     def run_async(self,x0,x1,autoInit=True,additionalOptions=None,jobid=None):
          if jobid is None: jobid = self.book_async()
          if autoInit:               self.buildInitGuess(x0,x1,jobid=jobid)
          return AcadoRunner.run_async(self,
                                       states={'init':  [ x0[:self.NQ].flat,x0[self.NQ:].flat ],
                                               'final': [ x1[:self.NQ].flat,x1[self.NQ:].flat ] },
                                       additionalOptions=additionalOptions,
                                       jobid = jobid)

     def join(self,jobid,x1=None,x2=None,threshold=1e-3,**kwargs):
          ret = AcadoRunner.join(self,jobid,*kwargs)
          return self.checkResult(ret,x1,x2,jobid,threshold)
          # if not ret: return False
          # if x1 is not None or x2 is not None: X = self.states()
          # if x1 is not None and norm(x1.T-X[0,:]) > self.threshold: return False
          # if x2 is not None and norm(x2.T-X[-1,:]) > self.threshold: return False
          # return True
          
     def initrun(self,x0,x1,iterations=100,autoInit = True):
          if autoInit:               self.buildInitGuess(x0,x1)
          return AcadoRunner.initrun(self,
                                     states={'init':  [ x0[:self.NQ].flat,x0[self.NQ:].flat ],
                                             'final': [ x1[:self.NQ].flat,x1[self.NQ:].flat ] },
                                     iterations=iterations)

     def rerun(self): return AcadoRunner.run(self)

     def states(self,jobid=None):
         '''The problem is Mayer-based, hence the cost is not part of the state file.'''
         return np.array(f2a(self.options['ostate']+self.async_ext(jobid)))[:,1:]

     def params(self,jobid=None):
         return np.array(f2a(self.options['oparam']+self.async_ext(jobid)))[:,1:]
     def times(self,jobid=None):
          '''Return times of state and control samplings.'''
          N = self.options['steps']
          return np.arange(0.,N+1)/N * self.opttime(jobid)
     def opttime(self,jobid=None):
          return self.params(jobid)[0,0]
     

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
    
