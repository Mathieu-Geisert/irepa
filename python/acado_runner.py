import os
import copy
import numpy as np

HORIZON  = 5.0
DT       = 0.25
NHORIZON = int(HORIZON/DT)

f2a = lambda filename: [ [float(s) for s in l.split() ] for l in open(filename).readlines()]

class AcadoRunner:
    def __init__(self,path= "/home/nmansard/src/pinocchio/pycado/build/unittest/discrete_pendulum"):
        self.exe = path
        self.dataroot = '/tmp/mpc'
        self.datactl  = '.ctl'
        self.datastx  = '.stx'
        self.controlFile  = self.dataroot + self.datactl
        self.stateFile    = self.dataroot + self.datastx
        self.options = { 
            'horizon'   : HORIZON,
            'steps'     : NHORIZON,
            'iter'      : 3,
            'icontrol'  : self.controlFile,
            'istate'    : self.stateFile,
            'ocontrol'  : self.controlFile,
            'ostate'    : self.stateFile,
            'shift'     : 1,
            }
        self.additionalOptions = ''
    def run(self,pos,vel,opts = None):
        if not opts: opts = self.options
        self.cmd = self.exe + ' ' \
            + ' '.join([ '--%s=%s' % (k,str(v)) for k,v in opts.items() ]) \
            + self.additionalOptions \
            + ' -p %.20f -v %.20f > /dev/null' % ( pos,vel )
        os.system(self.cmd)
        ctls = f2a(self.controlFile)
        stts = f2a(self.stateFile)
        return ctls[0][1:],stts[-1][-1]
    def initrun(self,pos,vel,i=100):
        opt = copy.copy(self.options)
        del opt['icontrol']
        del opt['istate']
        del opt['shift']
        opt['iter'] = i
        return self.run(pos,vel,opt)
        
    def states(self):
        return np.array(f2a(self.stateFile))[:,1:3]
    def controls(self):
        return np.array(f2a(self.controlFile))[:,1:2]
    def costs(self):
        return np.array(f2a(self.stateFile))[:,3]
