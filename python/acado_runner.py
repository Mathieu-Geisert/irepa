import os
import copy
import numpy as np

HORIZON  = 5.0
DT       = 0.25
NHORIZON = int(HORIZON/DT)

f2a = lambda filename: [ [float(s) for s in l.split() ] for l in open(filename).readlines()]
flatten = lambda matrx: np.array([matrx]).squeeze().tolist()

class AcadoRunner(object):
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
        self.verbose = False
        self.warningCodes = [
            121, # RET_MAX_NUMBER_OF_STEPS_EXCEEDED
            ]
    @property
    def iter(self): return self.options['iter']
    @iter.setter
    def iter(self,i):  self.options['iter'] = i

    @property
    def name(self):
        print 'totot'
        return 'r'
    @name.setter
    def name(self, value): print 'atata'
    
    def run(self,pos=None,vel=None,opts = None):
        if not opts: opts = self.options
        if pos is not None:
            opts['initpos'] = ' '.join([ '%.20f'%f for f in pos ])
        if vel is not None:
            opts['initvel'] = ' '.join([ '%.20f'%f for f in vel ])

        tostr = lambda s: '='+str(s) if s is not None else ''
        self.cmd = self.exe + ' ' \
            + ' '.join([ '--%s%s' % (k,tostr(v)) for k,v in opts.items() ]) \
            + self.additionalOptions
        if not self.verbose: 
            self.cmd += ' > /dev/null'
        self.retcode = os.system(self.cmd) >> 8
        if self.retcode is not 0 and self.retcode not in self.warningCodes:
            raise  RuntimeError("Error when executing Acado")
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
        
    def debug(self,reset=True):
        if reset:
            self.verbose = True
            self.options['plot'] = None
            self.options['printlevel'] = 2
        else:
            self.verbose = False
            try:
                del self.options['plot']
                del self.options['printlevel']
            except:
                pass


    def states(self):
        return np.array(f2a(self.stateFile))[:,1:-1]
    def controls(self):
        return np.array(f2a(self.controlFile))[:,1:]
    def costs(self):
        return np.array(f2a(self.stateFile))[:,-1]

