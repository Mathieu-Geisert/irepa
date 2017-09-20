from pinocchio.utils import *
import pinocchio as se3
from pendulum import Visual


class Quadcopter:
    def __init__(self, mass=2.5, length=.5, guess=1, withDisplay=False):

        self.mass = mass
        self.length = length
        self.guess = guess
        self.inertiaXX = 1.  # mass*length**2/3
        self.inertiaYY = 1.
        self.inertiaZZ = 1.6

        self.DT = 1e-2
        self.NDT = 20

        self.nx = 10
        self.nq = 5
        self.nv = 5
        self.nu = 4

        self.g = 9.81

        # These bounds are used for sampling new configurations.
        self.qup = np.matrix([2., 2., 2., np.pi/2, np.pi/2]).T
        self.qlow = -self.qup
        self.vup = np.matrix([2., 2., 2., 2., 2.]).T
        self.vlow = -self.vup
        # self.xup = np.vstack([self.qup, self.vup])
        # self.xlow = np.vstack([self.qlow, self.vlow])

        # These bounds are used for learning.
        self.qmax = np.matrix([5., 5., 5., np.pi/2, np.pi/2]).T
        self.vmax = np.matrix([5]*5).T
        self.xmax = np.concatenate([self.qmax,self.vmax])
        self.xmin = -self.xmax

        self.umax = np.matrix([mass * 5] * 4).T *2 # DEBUG
        self.umin = zero(4)

        self.withSinCos = False

        self.visuals = []
        self.initDisplay(withDisplay)

    @property
    def xup(self): return  np.vstack([self.qup, self.vup])
    @property
    def xlow(self): return  np.vstack([self.qlow, self.vlow])

    @property
    def nobs(self):
        return 4 if self.withSinCos else 3

    def initDisplay(self, withDisplay=True):
        if withDisplay:
            from display import Display
            self.viewer = Display()


            # color = [red, green, blue, transparency] = [1, 1, 0.78, 1.0]
            color1 = [1., 0., 0., 1.]
            color2 = [0., 1., 0., 1.]
            colorMot = [0., 0., 1., 1.]

            try:
                self.viewer.viewer.gui.createGroup('world/qc')
                self.viewer.viewer.gui.addCapsule('world/qc/body1', self.length * .1, 1.05 * self.length, color1)
                self.viewer.viewer.gui.applyConfiguration('world/qc/body1', [0., 0., 0., 0.707, 0., 0.707, 0.])
                self.viewer.viewer.gui.addCapsule('world/qc/body2', self.length * .1, 1.05 * self.length, color2)
                self.viewer.viewer.gui.applyConfiguration('world/qc/body2',[0., 0., 0., 0.707, 0.707, 0., 0.])
                self.viewer.viewer.gui.addCylinder('world/qc/motor1', self.length * 0.1, self.length * .2, colorMot)
                self.viewer.viewer.gui.applyConfiguration('world/qc/motor1', [self.length / 2., 0., 0., 1., 0., 0., 0.])
                self.viewer.viewer.gui.addCylinder('world/qc/motor2', self.length * 0.1, self.length * .2, colorMot)
                self.viewer.viewer.gui.applyConfiguration('world/qc/motor2', [-self.length / 2., 0., 0., 1., 0., 0., 0.])
                self.viewer.viewer.gui.addCylinder('world/qc/motor3', self.length * 0.1, self.length * .2, colorMot)
                self.viewer.viewer.gui.applyConfiguration('world/qc/motor3', [0., self.length / 2., 0., 1., 0., 0., 0.])
                self.viewer.viewer.gui.addCylinder('world/qc/motor4', self.length * 0.1, self.length * .2, colorMot)
                self.viewer.viewer.gui.applyConfiguration('world/qc/motor4', [0., -self.length / 2., 0., 1., 0., 0., 0.])
                self.viewer.viewer.gui.refresh()
                self.viewer.viewer.gui.addLandmark("world",1.)
            except:
                pass
            self.visuals.append(Visual('world/qc', 1, se3.SE3(rotate('y', 0.), zero(3))))

        else:
            self.viewer = None

    def sample(self):
        # return np.diagflat(self.xup-self.xlow)*rand(self.nx)+self.xlow
        q = np.diagflat(self.qup - self.qlow) * rand(self.nq) + self.qlow

        r = lambda a, b: np.random.normal((a + b) / 2., (b - a) / 6.)
        vs = []
        for iv in range(self.nv):
            while True:
                v = r(self.vlow.flat[iv], self.vup.flat[iv])
                if v >= self.vlow.flat[iv] and v <= self.vup.flat[iv]:
                    vs.append(v)
                    break

        # v = np.random.normal( (self.vup+self.vlow)/2,  np.sqrt((self.vup-self.vlow)/2) )
        # v = np.clip(v,self.vlow,self.vup)

        return np.concatenate([q, np.matrix(vs).T])

    def reset(self, x0=None):
        if x0 is None: x0 = self.sample()
        assert (len(x0) == self.nx)
        self.x = x0.copy()
        return self.obs(x0)

    def step(self, u):
        self.x = self.dynamics(self.x, u)

    def obs(self, x):
        assert (len(x) == self.nx)
        return np.hstack([x[:2], np.matrix([np.cos(x[2]), np.sin(x[2])])]) if self.withSinCos else x

    def display(self, x):
        x = x.flat
        M = se3.SE3(rotate('y', x[2]), np.matrix([x[0], 0., x[1]]).T)
        for v in self.visuals:
            v.place(self.viewer, M)
        self.viewer.viewer.gui.refresh()

    def dynamics(self, x, u):
        print 'NOT IMPLEMENTED YET...'
        assert 1==2
        withMatrix = isinstance(x, np.matrix)
        if withMatrix:
            assert (isinstance(u, np.matrix))
            x = np.array(x.flat)
            u = np.array(u.flat)
        else:
            x = x.copy()

        m = self.mass
        l = self.length
        I = self.inertia

        for _ in range(self.NDT):
            th = x[2]
            qdd = np.array([
                -1 / m * (u[0] + u[1]) * np.sin(th),
                +1 / m * (u[0] + u[1]) * np.cos(th) - self.g,
                l / I * (u[0] - u[1])
            ])
            qd = x[3:] + qdd * self.DT / self.NDT
            q = x[:3] + qd * self.DT / self.NDT
            x = np.concatenate([q, qd])

        if withMatrix: x = np.matrix(x).T
        return x

    def initialGuess(self, x0, x1, options):
        if self.guess == 1:
            #Quasistatic initial guess
            Ustat = self.mass * self.g / 4.
            U = np.ones([4, self.NDT])*Ustat
            X = np.zeros([self.nx, self.NDT])
            for i in range(self.NDT):
                X[0:3, i] = (1. - (float(i) / float(self.NDT - 1))) * x0[0:3].T + float(i) / float(self.NDT - 1) * x1[0:3].T
                X[5:8, i] = (1. - (float(i) / float(self.NDT - 1))) * x0[5:8].T + float(i) / float(self.NDT - 1) * x1[5:8].T

            T = np.arange(self.NDT+1)/float(self.NDT)
            return X, U, T
        else:
            assert False, "No other initial guess implemented than 1"

if __name__ == '__main__':
    env = Quadcopter(withDisplay=True)
