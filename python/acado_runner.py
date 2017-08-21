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
    
    def stateToString(self,label,p,v):
        cmd = ''
        if p is not None: cmd += ' --'+label+'pos='+' '.join([ '%.20f'%f for f in p ])
        if v is not None: cmd += ' --'+label+'vel='+' '.join([ '%.20f'%f for f in v ])
        return cmd

    def generateCommandLine(self,states,opts,additionalOptions=None):
        if not opts: opts = self.options
        if additionalOptions is None: additionalOptions = self.additionalOptions

        tostr = lambda s: '='+str(s) if s is not None else ''
        cmd = self.exe + ' ' \
            + ' '.join([ '--%s%s' % (k,tostr(v)) for k,v in opts.items() ])
        for label,[p,v] in states.items():
            cmd += self.stateToString(label,p,v)
        cmd += additionalOptions
        if not self.verbose:    cmd += ' > /dev/null'
        return cmd        

    def run(self,states=None,pos=None,vel=None,opts = None,additionalOptions = None):
        '''
        States is a dictionary containing label->[pos,vel] map, typically { 'init': [p0,v0] }.
        pos,vel are kept for compatibility. If present, they are added to the states dict 
        as 'init' lavel.
        '''
        if states is None: states = {}
        if pos is not None and vel is not None: states['init'] = [pos,vel]
        self.cmd = self.generateCommandLine(states,opts,additionalOptions)
        self.retcode = os.system(self.cmd) >> 8
        self.raiseErrorFromReturnCode(self.retcode)

    def setup_async(self,nbprocess = 32):
        import multiprocessing
        self.pool = multiprocessing.Pool(processes=nbprocess)
        self.nbprocess = nbprocess
        self.jobReferences = { i: None for i in range(nbprocess) }
        self.availableJobs = list(reversed(range(nbprocess)))
        
    def async_ext(self,jobid=None):
        return '' if jobid is None else '.%03d'%jobid
    def book_async(self):
        return self.availableJobs.pop()

    def run_async(self,states,opts = None, additionalOptions=None,jobid=None):
        '''
        Run Acado asynchronously. See join for getting the result.
        The method returns the reference to the job.
        '''
        if additionalOptions is None: additionalOptions = self.additionalOptions
        if jobid is None: jobid = self.book_async()
        assert( jobid not in self.availableJobs )
        cmd = self.generateCommandLine(states,opts,
                                       additionalOptions=additionalOptions+' --jobid='+self.async_ext(jobid))
        self.jobReferences[jobid] = self.pool.apply_async(os.system,(cmd,))
        self.cmd = cmd  # for debug only
        return jobid
        
    # def jobIdFromJobHandler(self,h):
    #     return next((k for k,v in self.jobReferences.items() if v == h), None)

    def join(self,jid,timeout=None,dorelease=True):
        '''Join the process (with timeout) and return the result returned by the process.'''
        jh = self.jobReferences[jid]
        retCode = jh.get(timeout=timeout)
        if dorelease: self.release(jid)
        return self.boolFromReturnCode(retCode)
    def release(self,jid):
        '''Optionnally, the release of the process (including possible overwrite of the state/control
        acado files containing the results) can be post-poned when joining the process. In that
        case, the user should manually call release.'''
        self.jobReferences[jid] = None
        self.availableJobs.append(jid)
        
    def joinAll(self,timeout=0.):
        for jid in range(self.nbprocess):
            if jid not in self.availableJobs:
                self.join(jid,timeout=timeout)

    def boolFromReturnCode(self,retcode):
        return retcode is 0 or retcode in self.warningCodes
    def raiseErrorFromReturnCode(self,retcode):
        if not self.boolFromReturnCode(retcode):
            raise  RuntimeError("Error when executing Acado")

        # ctls = f2a(self.controlFile)
        # stts = f2a(self.stateFile)
        # return ctls[0][1:],stts[-1][-1]


    def initrun(self,states,pos=None,vel=None,iterations=100,additionalOptions=None):
        opt = copy.copy(self.options)
        del opt['icontrol']
        del opt['istate']
        del opt['shift']
        opt['iter'] = i
        return self.run(states,pos,vel,opts=opt,additionalOptions=additionalOptions)
        
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


    def states(self,jobid=None):
        return np.array(f2a(self.options['ostate']+self.async_ext(jobid)))[:,1:-1]
    def controls(self,jobid=None):
        return np.array(f2a(self.options['ocontrol']+self.async_ext(jobid)))[:,1:]
    def costs(self,jobid=None):
        return np.array(f2a(self.options['ostate']+self.async_ext(jobid)))[:,-1]

