from pendulum import Pendulum
from scipy.optimize import *
from pinocchio.utils import *
import pinocchio as se3
import numpy as np
from numpy import sin,cos
from numpy.linalg import norm
import time
import signal
import matplotlib.pyplot as plt
import acado_runner
plt.ion()

env = Pendulum(1)


acado = acado_runner.AcadoRunner()
#acado.options['horizon'] = HORIZON  = 5.0
#acado.options['steps']   = NHORIZON = int(HORIZON/env.DT)

env.DT              = .25
env.NDT             = 1
env.Kf              = 0.1

database = []

with_plot = 1
with_render = 1

for rollout in range(1):
    env.reset(np.matrix([2.3,0.]).T)
    env.render()
    if not (rollout%10): 
        print 'Computing the initial trajectory <%d> from '%rollout,env.x.T
        time.sleep(2)
    acado.initrun(env.x[0,0],env.x[1,0])
    if with_plot:
        print acado.cmd
        print 'Trajectory computed'
        raw_input("Press Enter to continue...")

    # Initial optim
    for controlcycle in range(50):
        u,cost = acado.run(env.x[0,0],env.x[1,0])
        
        database.append( [ env.x[0,0], env.x[1,0], u[0], cost ] )
        env.step(np.matrix(u))
        if with_render: 
            env.render()
            print "U%d = %.3f\t\t%.1f" % (controlcycle,u[0],cost)
        if norm(env.x)<1e-4: break
        if with_plot:
            plt.plot(range(controlcycle,controlcycle+21),acado.states()[:,0])
            plt.plot([controlcycle,controlcycle],[-10,10])
            print acado.cmd
            raw_input("Press Enter to continue...")
            plt.figure(1).axes[0].lines.pop()

D=np.array(database)
import datetime
filename = 'database_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S')+'.np'
np.save(open(filename,'w'),D)
