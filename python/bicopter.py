from pinocchio.utils import *
import pinocchio as se3
from pendulum import Visual

class Bicopter:
    def __init__(self, mass=2.5, length=.5, withDisplay=True):

        self.mass    = mass
        self.length  = length
        self.inertia = 1. # mass*length**2/3

        self.DT      = 1e-2
        self.NDT     = 10

        self.nx      = 6
        self.nq      = 3
        self.nv      = 3
        self.nu      = 2

        self.g       = 9.81

        self.qup     = np.matrix([ 1., 1., np.pi/2 ]).T
        self.qlow    = -self.qup
        self.vup     = np.matrix([ 2., 2., 2. ]).T
        self.vlow    = -self.vup
        self.xup     = np.vstack([self.qup,  self.vup ])
        self.xlow    = np.vstack([self.qlow, self.vlow])

        self.umax    = np.matrix([ mass*10, ]*2).T
        self.umin    = zero(2)-1

        self.withSinCos = False

        self.visuals = []
        self.initDisplay(withDisplay)

        
    @property
    def nobs(self): return 4 if self.withSinCos else 3

    def initDisplay(self,withDisplay = True):
        if withDisplay:
            from display import Display
            self.viewer     = Display()
 
            color   = [red,green,blue,transparency] = [1,1,0.78,1.0]
            colorred = [1.0,0.0,0.0,1.0]

            try:self.viewer.viewer.gui.addCapsule('world/bcbody', self.length*.1,1.05*self.length,color)
            except:pass
            self.visuals.append( Visual('world/bcbody',1,se3.SE3(rotate('y',np.pi/2),zero(3))) )

            try:self.viewer.viewer.gui.addCylinder('world/bcmotor1', self.length*0.1,self.length*.2,colorred)
            except:pass
            self.visuals.append( Visual('world/bcmotor1',1,
                                        se3.SE3(eye(3),
                                                np.matrix([self.length/2.,0,self.length*.1]).T)) )

            try:self.viewer.viewer.gui.addCylinder('world/bcmotor2', self.length*0.1,self.length*.2,colorred)
            except:pass
            self.visuals.append( Visual('world/bcmotor2',1,
                                        se3.SE3(eye(3),
                                                np.matrix([-self.length/2.,0,self.length*.1]).T)) )

            try:self.viewer.viewer.gui.addCylinder('world/bcprop1',self.length/3,self.length/50,color)
            except:pass
            self.visuals.append( Visual('world/bcprop1',1,
                                        se3.SE3(eye(3),
                                                np.matrix([ self.length/2.,0,self.length*.2]).T)) )
   
            try:self.viewer.viewer.gui.addCylinder('world/bcprop2',self.length/3,self.length/50,color)
            except:pass
            self.visuals.append( Visual('world/bcprop2',1,
                                        se3.SE3(eye(3),
                                                np.matrix([-self.length/2.,0,self.length*.2]).T)) )

        else:
            self.viewer = None

    def sample(self):
        #return np.diagflat(self.xup-self.xlow)*rand(self.nx)+self.xlow
        q = np.diagflat(self.qup-self.qlow)*rand(self.nq)+self.qlow

        r = lambda a,b: np.random.normal( (a+b)/2., (b-a)/6. )
        vs = []
        for iv in range(self.nv):
            while True:
                v = r(self.vlow.flat[iv],self.vup.flat[iv])
                if v>=self.vlow.flat[iv] and v<=self.vup.flat[iv]: 
                    vs.append(v)
                    break

        #v = np.random.normal( (self.vup+self.vlow)/2,  np.sqrt((self.vup-self.vlow)/2) )
        #v = np.clip(v,self.vlow,self.vup)
        
        return np.concatenate([q,np.matrix(vs).T])

    def reset(self,x0 = None):
        if x0 is None: x0 = self.sample()
        assert(len(x0) == self.nx)
        self.x = x0.copy()
        return self.obs(x0)

    def step(self,u):
        self.x = self.dynamics(self.x,u)

    def obs(self,x):
        assert(len(x) == self.nx)
        return np.hstack([ x[:2], np.matrix([ np.cos(x[2]), np.sin(x[2]) ]) ]) if self.withSinCos else x

    def display(self,x):
        x = x.flat
        M = se3.SE3(rotate('y',x[2]), np.matrix([ x[0], 0., x[1] ]).T )
        for v in self.visuals: 
            v.place(self.viewer,M)
        self.viewer.viewer.gui.refresh()

    def dynamics(self,x,u):
        withMatrix = isinstance(x,np.matrix)
        if withMatrix:
            assert(isinstance(u,np.matrix))
            x = np.array(x.flat)
            u = np.array(u.flat)
        else: x = x.copy()

        m = self.mass
        l = self.length
        I = self.inertia

        for _ in range(self.NDT):
            th = x[2]
            qdd = np.array([ 
                    -1/m*(u[0]+u[1])*np.sin(th),
                     +1/m*(u[0]+u[1])*np.cos(th)-self.g,
                     l/I*(u[0]-u[1])
                     ])
            qd  = x[3:] + qdd*self.DT/self.NDT
            q   = x[:3] + qd *self.DT/self.NDT
            x   = np.concatenate([q,qd])

        if withMatrix: x=np.matrix(x).T
        return x

if __name__ == '__main__':
    env = Bicopter(withDisplay=True)
