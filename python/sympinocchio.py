from sympy import symbols,cos,sin
from sympy.interactive.printing import init_printing
init_printing(use_unicode=False, wrap_line=False, no_global=True)
import sympy.matrices as sm
from sympy.matrices import Matrix, eye, zeros, ones, diag
from sympy.physics.mechanics.functions import cross
from sympy.physics.mechanics import ReferenceFrame, Vector
import numpy as np
import pinocchio.utils as ut

p,m,g = symbols('p m g')

class SE3:
    def __init__(self,R=None,p=None):
        self.R = R if R is not None else Matrix([[ 1.,0.,0. ],[ 0.,1.,0. ],[ 0.,0.,1. ]])
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


Ycst = Inertia(mass=m,lever=Matrix([ 0.,0.,p/2 ]) ) 
JOINTS = [ 
    Joint(0), ### Universe
    Joint(parent=0,Mfix=SE3(),                        inertia = Ycst),
    Joint(parent=1,Mfix=SE3( p=Matrix([ 0.,0.,p ]) ), inertia = Ycst)
    ]

q = [ symbols("q%d"%j)  for j,_ in enumerate(JOINTS[1:]) ]
vq = [ symbols("dq%d"%j) for j,_ in enumerate(JOINTS[1:]) ]

class RNEA:
    def __init__(self,kinetree):
        self.joints = kinetree
        self.njoints= len(kinetree)
        self.oMi    = [ SE3() for j in self.joints ]
        self.liMi   = [ SE3() for j in self.joints ]
        self.v      = [ Motion() for j in self.joints ]
        self.a      = [ Motion() for j in self.joints ]
        self.f      = [ Force() for j in self.joints ]
        self.Ycrb   = [ Inertia() for j in self.joints ]
        self.tauq   = [ 0       for j in self.joints[1:] ]
        self.M      = Matrix([ [0,]*(self.njoints-1), ]*(self.njoints-1))

        self.a[0]   = Motion(v=Matrix([0,0,g]))

    def fkine(self,q,vq):
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

    def back(self):
        for j,joint in reversed(list(enumerate(self.joints))):
            if j==0: continue
            print j
            self.tauq[j-1]  = joint.Si.T*self.f[j]
            self.f[j-1]    += self.liMi[j] * self.f[j]

    def __call__(self,q,vq):
        self.fkine(q,vq)
        self.back()
        return self.tauq

    def crba(self):
        # Backward loop of crba
        for j,joint in reversed(list(enumerate(self.joints))):
            if j==0: continue
            self.Ycrb[j-1] = self.Ycrb[j-1] + self.liMi[j]*self.Ycrb[j]
            YS = self.oMi[j]*(self.Ycrb[j]*joint.Si)
            for jj in range(j,0,-1):
                print "jj = ",jj
                Sjj = self.oMi[jj]*self.joints[jj].Si
                self.M[jj-1,j-1] = Sjj.T*YS
                if jj is not j: self.M[j-1,jj-1] = Sjj.T*YS

rnea = RNEA(JOINTS)
rnea(q,vq)
rnea.crba()


def values(valq,valv):
    return { m:1,p:.1,g:9.81,
             vq[0]:valv[0,0] ,vq[1]:valv[1,0],
             q[0] :valq[0,0] ,q[1] :valq[1,0]
             }


