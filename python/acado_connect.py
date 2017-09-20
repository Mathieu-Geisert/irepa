from pinocchio.utils import *
import pinocchio as se3
from time import sleep
from numpy.linalg import inv,norm
from acado_runner import *
from acado_runner import AcadoRunner


class InitGuessBuilder:
     def __init__(self,model=None,data=None):
          self.model = model
          if self.model is not None: 
               self.data = model.createData()
               self.NQ   = model.nq
               self.NV   = model.nv
               self.NX   = self.NQ+self.NV

     def __call__(self,x0,x1,options):
          T      = options['horizon']
          NSTEPS = options['steps']
          DT     = T / NSTEPS
          NQ,NV,NX = self.NQ,self.NV,self.NX

          withControl = self.model is not None and self.data is not None
          assert(NQ == NV)

          # try:
          # Quasistatic initial guess
          withControl = True
          Ustat = 2.5 * 9.81 / 4.
          U = np.ones([4, NSTEPS+1]) * Ustat
          X = np.zeros([NX, NSTEPS+1])
          for i in range(NSTEPS+1):
               X[0:3, i] = (1. - (float(i) / float(NSTEPS))) * x0[0:3].T + float(i) / float(NSTEPS) * x1[0:3].T
               X[5:8, i] = (1. - (float(i) / float(NSTEPS))) * x0[5:8].T + float(i) / float(NSTEPS) * x1[5:8].T

          T = np.arange(NSTEPS + 1) / float(NSTEPS)
          # X = np.zeros([NX, NSTEPS+1])
          return X.T, U.T, T
          # except:
          N = 3                   # Polynom degree
          C = zero([4,N+1])       # Matrix of t**i coefficients
          b = zero(4)             # Vector of x,xdot references

          t = 0.0
          C[0,:] =  [ t**i for i in range(N+1) ]
          C[1,1:] = [ i*t**(i-1) for i in range(1,N+1) ]
          C[2,:] =  [ T**i for i in range(N+1) ]
          C[3,1:] = [ i*T**(i-1) for i in range(1,N+1) ]

          P = []
          V = []
          A = []
          U = []
          for iq in range(NQ):

               b[:2] = x0[iq::NQ]
               b[2:] = x1[iq::NQ]

               c = inv(C)*b

               P.append( np.vstack([ [t**i \
                                           for i in range(  N+1)]*c     for t in np.arange(0,T+DT/2,DT) ]) )
               V.append (np.vstack([ [i*t**(i-1) \
                                           for i in range(1,N+1)]*c[1:] for t in np.arange(0,T+DT/2,DT) ]))
               A.append( np.vstack([ [i*(i-1)*t**(i-2) \
                                           for i in range(2,N+1)]*c[2:] for t in np.arange(0,T+DT/2,DT) ]))

          X = np.hstack(P+V)

          if 'armature' in options:
               armature = options['armature']
               dyninv = lambda p,v,a: se3.rnea(self.model,self.data,p,v,a)+armature*a
          else:
               dyninv = lambda p,v,a: se3.rnea(self.model,self.data,p,v,a)

          U = np.vstack([ dyninv(p.T,v.T,a.T).T for p,v,a in zip(np.hstack(P),np.hstack(V),np.hstack(A)) ]) \
              if self.model is not None else None

          # # while horizon T is optimized, timescale should be rescaled between 0 and 1
          # # see http://acado.sourceforge.net/doc/html/d4/d29/example_002.html
          T = np.arange(NSTEPS+1)*T/NSTEPS
          print X.shape, T.shape
          return X,U,T



class AcadoConnect(AcadoRunner):
     def __init__(self,path="/home/nmansard/src/pinocchio/pycado/build/unittest/connect_pendulum",
                  model=None,data=None,datadir='/tmp/'):

          AcadoRunner.__init__(self,path)

          self.options['istate']   = datadir+'guess.stx'
          self.options['icontrol'] = datadir+'guess.ctl'
          self.options['ocontrol'] = datadir+'mpc.ctl'
          self.options['ostate']   = datadir+'mpc.stx'
          self.options['oparam']   = datadir+'mpc.prm'

          self.withRunningCost     = False     # The problem is Mayer cost.

          self.guess = InitGuessBuilder(model,data)

     def setDims(self,NQ=None,NV=None):
          if NQ is None: NQ=self.NQ
          if NV is None: NV=self.NV
          self.NQ       = NQ
          self.NV       = NV
          self.guess.NQ = NQ
          self.guess.NV = NV
          self.guess.NX = NQ+NV

     # --- INIT GUESS ---------------------------------------------------------------------
     # --- INIT GUESS ---------------------------------------------------------------------
     # --- INIT GUESS ---------------------------------------------------------------------

     def buildInitGuess(self,x0,x1,jobid=None):
          X,U,T = self.guess(x0,x1,self.options)

          # while horizon T is optimized, timescale should be rescaled between 0 and 1 
          # see http://acado.sourceforge.net/doc/html/d4/d29/example_002.html

          if 'istate' in self.options:
               np.savetxt(self.stateFile('i',jobid),np.vstack([T/T[-1], X.T]).T)
          if U is not None and 'icontrol' in self.options:
               np.savetxt(self.controlFile('i',jobid),np.vstack([T/T[-1], U.T]).T)

          return X,U,T

     # --- RUN WITH INIT GUESS ------------------------------------------------------------
     # --- RUN WITH INIT GUESS ------------------------------------------------------------
     # --- RUN WITH INIT GUESS ------------------------------------------------------------
     def stateDict(self,x0,x1):
          return {'init':  [ x0[:self.NQ].flat,x0[self.NQ:].flat ],
                  'final': [ x1[:self.NQ].flat,x1[self.NQ:].flat ] }

     def run(self,x0,x1,autoInit=True,threshold=1e-3,**kwargs):
          if autoInit:               
               _,_,T = self.buildInitGuess(x0,x1)
               if 'horizon' not in self.options:
                    if 'additionalOptions' not in kwargs: kwargs['additionalOptions'] = ''
                    kwargs['additionalOptions'] +=  ' --horizon=%.4f' % T[-1]
          AcadoRunner.run(self,states=self.stateDict(x0,x1), **kwargs)
          # # Run may raise an error but always return True.
          return self.checkResult(True,x0,x1,threshold=threshold)

     def checkResult(self,ret,x0=None,x1=None,jobid=None,threshold=1e-3):
          if not ret: return False
          if 'Tmin' in self.options and self.opttime(jobid)<self.options['Tmin']:  return False
          if self.opttime(jobid)<0:  return False
          if x0 is not None or x1 is not None: X = self.states(jobid)
          if x0 is not None and norm(x0.T-X[ 0,:]) > threshold: return False
          if x1 is not None and norm(x1.T-X[-1,:]) > threshold: return False
          return True

     def initrun(self,x0,x1,iterations=100,autoInit = True,**kwargs):
          if autoInit:               self.buildInitGuess(x0,x1)
          AcadoRunner.initrun(self,
                              states=self.stateDict(x0,x1),
                              iterations=iterations)
          return self.checkResult(True,x0,x1)

     def rerun(self): return AcadoRunner.run(self)   # DEPREC???

     # --- ASYNC --------------------------------------------------------------------------
     # --- ASYNC --------------------------------------------------------------------------
     # --- ASYNC --------------------------------------------------------------------------
     def run_async(self,x0,x1,autoInit=True,jobid=None, **kwargs):
          # We have to pre-book to know where the init-guess should be stored.
          if jobid is None: jobid = self.book_async()
          if autoInit:
               _,_,T = self.buildInitGuess(x0,x1,jobid=jobid)
               if 'horizon' not in self.options:
                    if 'additionalOptions' not in kwargs: kwargs['additionalOptions'] = ''
                    kwargs['additionalOptions'] +=  ' --horizon=%.4f' % T[-1]
          return AcadoRunner.run_async(self,
                                       states = self.stateDict(x0,x1),
                                       jobid  = jobid, **kwargs)

     def join(self,jobid,x1=None,x2=None,threshold=1e-3,**kwargs):
          '''Join the solver process and check if the result satisfies the constraints'''
          ret = AcadoRunner.join(self,jobid,**kwargs)
          return self.checkResult(ret,x1,x2,jobid,threshold)
               

### --- UNIT TEST ---
### --- UNIT TEST ---
### --- UNIT TEST ---
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
    
