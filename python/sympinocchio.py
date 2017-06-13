from sympy import symbols,cos,sin,fraction
from sympy.interactive.printing import init_printing
init_printing(use_unicode=False, wrap_line=False, no_global=True)
import sympy.matrices as sm
from sympy.matrices import Matrix, eye, zeros, ones, diag
from sympy.physics.mechanics.functions import cross
from sympy.physics.mechanics import ReferenceFrame, Vector
import numpy as np
import pinocchio.utils as ut


class SE3:
    def __init__(self,R=None,p=None):
        self.R = R if R is not None else Matrix([[ 1,0.,0. ],[ 0.,1,0. ],[ 0.,0.,1 ]])
        self.p = p if p is not None else Matrix([ 0.,0.,0. ])
    def mult_se3(self,M):
        return SE3(self.R*M.R,self.p+self.R*M.p)
    def mult_inertia(self,I):
        return Inertia(I.m,self.p+self.R*I.c,self.R*I.I*self.R.T)  ## TODO
    def mult_motion(self,nu):
        return Motion(v=self.R*nu.v+self.p.cross(self.R*nu.w),
                      w=self.R*nu.w) 
    def mult_force(self,phi):
        return Force(f=self.R*phi.f,
                     t=self.R*phi.t+self.p.cross(self.R*phi.f))
    def __mul__(self,x):
        if    isinstance(x,SE3):    return self.mult_se3(x)
        elif isinstance(x,Inertia): return self.mult_inertia(x)
        elif isinstance(x,Motion):  return self.mult_motion(x)
        elif isinstance(x,Force):   return self.mult_force(x)
        else: raise TypeError('Argument SE3.__mul__ does not have the proper type: '+str(x.__class__))
    def __repr__(self):
        return "R="+self.R.__repr__()+" ,  p="+self.p.__repr__()

    def act(self,x):        return self.__mul__(x)
    def imult_motion(self,nu):
        ''' R'(v-pxw), R'w    '''
        return Motion(v=self.R.T*(nu.v-self.p.cross(nu.w)),
                      w=self.R.T*nu.w) 
    def actInv(self,x):
        if isinstance(x,Motion):  return self.imult_motion(x)
        elif isinstance(x,Force):   return self.imult_force(x)
        else: raise TypeError('Argument SE3.__actinv__ does not have the proper type: '+str(x.__class__))
        
class Motion:
    def __init__(self,v=None,w=None):
        self.v = v if v is not None else Matrix([ 0.,0.,0. ])
        self.w = w if w is not None else Matrix([ 0.,0.,0. ])
    def sum_motion(self,nu):
        return Motion(v=self.v+nu.v,w=self.w+nu.w)
    def cross_motion(self,nu):
        '''v1xw2 + w1xv2, w1xw2'''
        return Motion(v=self.v.cross(nu.w)+self.w.cross(nu.v), w=self.w.cross(nu.w))
    def cross_force(self,phi):
        ''' wxf, wxt+vxf'''
        return Force(f=self.w.cross(phi.f), t=self.w.cross(phi.t)+self.v.cross(phi.f))
    @property
    def T(self): return MotionT(self)
    def __add__(self,x): 
        if isinstance(x,Motion): return self.sum_motion(x)
        else: raise TypeError('Argument Motion.__add__ does not have the proper type: '+str(x.__class__))
    def __xor__(self,x):
        if   isinstance(x,Motion): return self.cross_motion(x)
        elif isinstance(x,Force):  return self.cross_force(x)
        else: raise TypeError('Argument Motion.__xor__ does not have the proper type: '+str(x.__class__))
    def __repr__(self):
        return "v="+self.v.__repr__()+"  w="+self.w.__repr__()

class MotionT:
    def __init__(self,v): self.v=v
    def mult_force(self,phi): # scalar product
        return self.v.v.dot(phi.f)+self.v.w.dot(phi.t)
    def __mul__(self,x):
        if isinstance(x,Force): return self.mult_force(x)
        else: raise TypeError('Argument Force.__mul__ does not have the proper type: '+str(x.__class__))

class Force:
    def __init__(self,f=None,t=None):
        self.f = f if f is not None else Matrix([ 0.,0.,0. ])
        self.t = t if t is not None else Matrix([ 0.,0.,0. ])
    def sum_force(self,phi):
        return Force(f=self.f+phi.f,t=self.t+phi.t)
    def __add__(self,x): 
        if isinstance(x,Force): return self.sum_force(x)
        else: raise TypeError('Argument Force.__add__ does not have the proper type: '+str(x.__class__))
       
class Inertia:
    def __init__(self,mass=0.0,lever=None,inertia=None):
        self.m    = mass
        self.c    = lever   if lever   is not None else Matrix([ 0.,0.,0. ])
        self.I    = inertia if inertia is not None else Matrix([[ 0.,0.,0. ],[ 0.,0.,0. ],[ 0.,0.,0. ]])
    def sum_inertia(self,Y):
        # Y_{a+b} = ( m_a+m_b,
        #            (m_a*c_a + m_b*c_b ) / (m_a + m_b),
        #             I_a + I_b - (m_a*m_b)/(m_a+m_b) * AB_x * AB_x )
        AB = self.c-Y.c
        x=AB[0,0]
        y=AB[1,0]
        z=AB[2,0]
        ABx2 = Matrix([[    -y*y-z*z,   x*y    ,   x*z     ],
                       [     x*y    ,  -x*x-z*z,   y*z     ],
                       [     x*z    ,   y*z    ,  -x*x-y*y ]]);
        return Inertia(self.m + Y.m,
                       (self.m*self.c + Y.m*Y.c)/(self.m+Y.m),
                       self.I + Y.I - (self.m*Y.m)/(self.m+Y.m)*ABx2)
    def mult_motion(self,nu):
        '''mv - mcxw, mcxv + (I)w'''
        f3=self.m*(nu.v - self.c.cross(nu.w))
        return Force(f=f3,
                     t=self.c.cross(f3) + self.I*nu.w)
    def __add__(self,x): 
        if isinstance(x,Inertia): return self.sum_inertia(x)
        else: raise TypeError('Argument Inertia.__add__ does not have the proper type: '+str(x.__class__))
    def __mul__(self,x):
        if isinstance(x,Motion): return self.mult_motion(x)
        else: raise TypeError('Argument Inertia.__mul__ does not have the proper type: '+str(x.__class__))
    def __repr__(self):
        return "m="+self.m.__repr__()+"  c="+self.c.T.__repr__()+"  I="+self.I.__repr__()
    def copy(self):
        return Inertia(self.m,self.c.copy(),self.I.copy())

def RX(qi):
    ci = cos(qi); si = sin(qi)
    return SE3(R=Matrix( [[ 1,0,0], [ 0,ci,si], [0,-si,ci]] ))
def RY(qi):
    ci = cos(qi); si = sin(qi)
    return SE3(R=Matrix( [[ ci,0,si],[0,1,0],[-si,0,ci]] ))
def RZ(qi):
    ci = cos(qi); si = sin(qi)
    return SE3(R=Matrix( [[ ci,si,0], [-si,ci,0], [0,0,1]] ))

class Joint:
    def __init__(self,parent,Mfix=None,inertia=None):
        self.parent = parent
        self.Mfix = Mfix    if Mfix    is not None else SE3()
        self.Y    = inertia if inertia is not None else Inertia()
        self.Mi   = SE3()
        self.vi   = Motion()
    def calc(self,qi,vqi):
        self.Mi = RY(qi)
        self.vi = Motion(w=Matrix([0,vqi,0]))
        self.Si = Motion(w=Matrix([0,1  ,0]))


class SymModel:
    def __init__(self):
        self.joints = [ Joint(0), ] ### Universe

    def addJoint(self,parent,jointPlacement,inertia):
        '''Only RY joints are accepted. Remember to call initData after all joints have
        been added.'''
        j = len(self.joints)
        self.joints.append( Joint(parent,jointPlacement,inertia) )

    def initData(self):
        '''Init the data structure once all joints have been added. The method
        should be called after any new call to addJoint.'''
        self.oMi    = [ SE3() for j in self.joints ]
        self.liMi   = [ SE3() for j in self.joints ]
        self.v      = [ Motion() for j in self.joints ]
        self.a      = [ Motion() for j in self.joints ]
        self.f      = [ Force() for j in self.joints ]
        self.Ycrb   = [ Inertia() for j in self.joints ]
        self.tauq   = Matrix([ 0       for j in self.joints[1:] ])
        self.q      = [ symbols("q%d" %(j)) for j,_ in enumerate(self.joints[1:]) ]
        self.vq     = [ symbols("vq%d"%(j)) for j,_ in enumerate(self.joints[1:]) ]
        self.g      = symbols('g')

        self.a[0]   = Motion(v=Matrix([0,0,self.g]))  # Init acceleration to gravity free fall

    def forwardKine(self,q,vq):
        '''
        Compute foward geometry (oMi, liMi), fwd kinematics (vq), centrifugal and gravity acc
        aq, initialize Ycrb and f.
        '''
        assert( len(self.joints)==len(self.v) )
        for j,joint in enumerate(self.joints):
            if j==0: continue
            joint.calc(q[j-1],vq[j-1])

            self.liMi[j] = joint.Mfix * joint.Mi
            self.oMi[j] = self.oMi[j-1] * self.liMi[j]

            vj  = joint.vi
            self.v[j] = self.liMi[j].actInv(self.v[j-1]) + vj

            aj  = self.v[j] ^ vj
            self.a[j] = self.liMi[j].actInv(self.a[j-1]) + aj

            self.f[j] = joint.Y*self.a[j] + (self.v[j] ^ (joint.Y * self.v[j]))

            self.Ycrb[j] = joint.Y.copy()

    def rnea(self,q=None,vq=None,redoFwdKine=True): # Compute b(q,vq), i.e rnea(q,vq,aq=0)
        if q is None: q = self.q
        if vq is None: vq = self.vq
        if redoFwdKine: self.forwardKine(q,vq) # Forward loop
        for j,joint in reversed(list(enumerate(self.joints))):
            # Backward loop
            if j==0: continue
            self.tauq[j-1]  = joint.Si.T*self.f[j]
            self.f[j-1]    += self.liMi[j] * self.f[j]
        return self.tauq

    def crba(self,q=None,redoFwdKine=True):
        if q is None: q = self.q
        if redoFwdKine: self.forwardKine(q,self.vq) # Forward loop
        self.M      = Matrix([ [0,]*(len(self.joints)-1), ]*(len(self.joints)-1))
        for j,joint in reversed(list(enumerate(self.joints))):
        # Backward loop of crba
            if j==0: continue
            self.Ycrb[j-1] = self.Ycrb[j-1] + self.liMi[j]*self.Ycrb[j]
            YS = self.oMi[j]*(self.Ycrb[j]*joint.Si)
            for jj in range(j,0,-1):
                Sjj = self.oMi[jj]*self.joints[jj].Si
                self.M[jj-1,j-1] = Sjj.T*YS
                if jj is not j: self.M[j-1,jj-1] = Sjj.T*YS
        return self.M


if __name__ == '__main__':
    # Create double pendulum with p length and m mass, and compare it with pinocchio values.
    p,m,c = symbols('p m c')
    Ycst = Inertia(mass=m,lever=Matrix([ 0.,0.,c ]) ) 
    model = SymModel()
    model.addJoint(parent=0,jointPlacement=SE3(),                        inertia = Ycst),
    model.addJoint(parent=1,jointPlacement=SE3( p=Matrix([ 0.,0.,p ]) ), inertia = Ycst)
    model.initData()

    b = model.rnea()
    b.simplify()
    M = model.crba(redoFwdKine=False)
    M.simplify()
    a = symbols('a')
    A = eye(2)*a

    def values(valq,valv):
        '''
        Generate a symbol table for a specific set of numerical values.
        '''
        return { m:1,p:.1,c:.05,model.g:9.81,
                 model.vq[0]:valv[0,0] ,model.vq[1]:valv[1,0],
                 model.q[0] :valq[0,0] ,model.q[1] :valq[1,0]
                 }


    import pinocchio as se3
    from pinocchio.utils import *
    from pendulum import Pendulum
    env                 = Pendulum(2,length=.1)       # Continuous pendulum

    print 'Statistic assert of model correctness....'
    for i in range(100):
        q  = rand(2)
        vq = rand(2)
        bv=se3.rnea(env.model,env.data,q,vq,zero(2))
        Mv=se3.crba(env.model,env.data,q)
        assert( (b.subs(values(q,vq))-bv).norm()<1e-6 )
        assert( (M.subs(values(q,vq))-Mv).norm()<1e-6 )
    print '\t\t\t\t\t... ok'

    b = b.subs({c:p/2})
    b.simplify()

    Mi = (M+A).inv()
    Mi = Mi.subs({c:p/2})
    Mi.simplify()
    _,denom= fraction(Mi[0,0])
    denom.simplify()
    Mi = denom*Mi
    Mi.simplify()
