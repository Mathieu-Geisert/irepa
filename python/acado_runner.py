import os
import copy
import numpy as np

HORIZON  = 5.0
DT       = 0.25
NHORIZON = int(HORIZON/DT)

f2a = lambda filename: np.array([ [float(s) for s in l.split() ] for l in open(filename).readlines()])
flatten = lambda matrx: np.array([matrx]).squeeze().tolist()

class AcadoRunner(object):
    def __init__(self,path= "/home/nmansard/src/pinocchio/pycado/build/unittest/discrete_pendulum"):
        self.exe = path
        self.dataroot = '/tmp/mpc'
        self.datactl  = '.ctl'
        self.datastx  = '.stx'

        controlFile  = self.dataroot + self.datactl
        stateFile    = self.dataroot + self.datastx
        self.options = { 
            'horizon'   : HORIZON,
            'steps'     : NHORIZON,
            'iter'      : 3,
            'icontrol'  : controlFile,
            'istate'    : stateFile,
            'ocontrol'  : controlFile,
            'ostate'    : stateFile,
            'shift'     : 1,
            }
        self.additionalOptions = ''
        self.verbose = False
        self.warningCodes = [
            121, # RET_MAX_NUMBER_OF_STEPS_EXCEEDED
            ]
        self.withRunningCost = True

    # --- RUN ----------------------------------------------------------------------------
    # --- RUN ----------------------------------------------------------------------------
    # --- RUN ----------------------------------------------------------------------------
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

    def boolFromReturnCode(self,retcode):
        return retcode is 0 or retcode in self.warningCodes
    def raiseErrorFromReturnCode(self,retcode):
        if not self.boolFromReturnCode(retcode):
            raise  RuntimeError("Error when executing Acado")

    def initrun(self,states,pos=None,vel=None,iterations=100,additionalOptions=None):
        opt = copy.copy(self.options)
        del opt['icontrol']
        del opt['istate']
        del opt['shift']
        opt['iter'] = i
        return self.run(states,pos,vel,opts=opt,additionalOptions=additionalOptions)
        
    # --- ASYNC --------------------------------------------------------------------------
    # --- ASYNC --------------------------------------------------------------------------
    # --- ASYNC --------------------------------------------------------------------------

    def setup_async(self,nbprocess = 32,nbrequest = None):
        import multiprocessing
        if nbrequest is None: nbrequest = nbprocess

        self.pool = multiprocessing.Pool(processes=nbprocess)
        self.nbprocess = nbprocess
        self.jobReferences = { i: None for i in range(nbrequest) }
        self.availableJobs = list(reversed(range(nbrequest)))
        self.cmds = {}
        
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
        self.cmds[jobid] = cmd  # for debug only
        return jobid
        
    def join(self,jid,timeout=None,dorelease=True):
        '''Join the process (with timeout) and return the result returned by the process.'''
        jh = self.jobReferences[jid]
        retCode = jh.get(timeout=timeout) >> 8
        if dorelease: self.release(jid)
        return self.boolFromReturnCode(retCode)

    def release(self,jid):
        '''Optionnally, the release of the process (including possible overwrite of the state/control
        acado files containing the results) can be post-poned when joining the process. In that
        case, the user should manually call release.'''
        self.jobReferences[jid] = None
        self.availableJobs.append(jid)
        
    def joinAll(self,timeout=0.):
        for jid,jh in self.jobReferences.items():
            if jh is not None:
                assert( jid not in self.availableJobs )
                self.join(jid,timeout=timeout)


    # --- READ TRAJECTORY FILES ----------------------------------------------------------
    # --- READ TRAJECTORY FILES ----------------------------------------------------------
    # --- READ TRAJECTORY FILES ----------------------------------------------------------

    def stateFile  (self,io,jobid=None): 
        assert( io =="o" or io =="i")
        return self.options[io+'state']+self.async_ext(jobid)
    def controlFile(self,io,jobid=None): 
        assert( io =="o" or io =="i")
        return self.options[io+'control']+self.async_ext(jobid)
    def paramFile(self,io,jobid=None): 
        assert( io =="o" or io =="i")
        return self.options[io+'param']+self.async_ext(jobid)

    def states(self,jobid=None):
        X = f2a(self.stateFile('o',jobid))
        return X[:,1:-1] if self.withRunningCost else X[:,1:]
    def controls(self,jobid=None):
        return f2a(self.controlFile('o',jobid))[:,1:]
    def costs(self,jobid=None):
        assert(self.withRunningCost)
        return f2a(self.stateFile('o',jobid))[:,-1]

    def params(self,jobid=None):
        return f2a(self.paramFile('o',jobid))[:,1:]

    def opttime(self,jobid=None):
        # Here is hardcoded the index of the time in the param table.
        return self.params(jobid)[0,0]

    def times(self,jobid=None):
        '''Return times of state and control samplings.'''
        N = self.options['steps']
        return np.arange(0.,N+1)/N * self.opttime(jobid)
    def cost(self,jobid=None):
        if self.withRunningCost:        return self.costs(jobid)[-1]
        else:                           return self.opttime(jobid)


    # --- SHORTCUTS ----------------------------------------------------------------------
    # --- SHORTCUTS ----------------------------------------------------------------------
    # --- SHORTCUTS ----------------------------------------------------------------------
    @property
    def iter(self): return self.options['iter']
    @iter.setter
    def iter(self,i):  self.options['iter'] = i

    def setTimeInterval(self,T,T0=0.001):
          self.options['horizon'] = T
          self.options['Tmin'] = T0
          self.options['Tmax'] = T*4

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
    
