import collections
import importlib

class MFM:
  def __init__(self, tau, sfs, afs, cuda=False):
    ## Import Dynamic Modules
    self.load_modules(cuda)

    ## Variables to Self
    self.tau = tau
    self.afs = afs
    self.sfs = sfs
    self.dec = sfs/afs

    ## Make De-emphasis Filter
    x = self.np.exp(-1/(self.afs * self.tau))
    self.db = [1-x]; self.da = [1,-x]
    self.zi = self.ss.lfilter_zi(self.db, self.da)

    ## Setup continuity data
    self.co = {
      "dc": collections.deque(maxlen=32),
      "diff": self.xp.array([0.0]),
    }

  def load_modules(self, cuda):
    self.cuda = cuda

    if self.cuda:
      self.xs = importlib.import_module('cusignal')
      self.xp = importlib.import_module('cupy')
      self.np = importlib.import_module('numpy')
      self.ss = importlib.import_module('scipy.signal')
    else:
      self.xs = importlib.import_module('scipy.signal')
      self.xp = importlib.import_module('numpy')
      self.np = self.xp 
      self.ss = self.xs

  def run(self, buff):
    b = self.xp.array(buff)
    d = self.xp.concatenate((self.co['diff'], self.xp.angle(b)), axis=None)
    b = self.xp.diff(self.xp.unwrap(d))
    self.co['diff'][0] = d[-1]
    b /= self.xp.pi

    # Normalize for DC
    dc = self.xp.mean(b)
    self.co['dc'].append(dc)
    b -= self.np.mean(self.co['dc'])
  
    # Demod Left + Right (LPR)
    LPR = self.xs.resample_poly(b, 1, int(self.sfs//self.afs), window='hamm')
    LPR, self.zi = self.xs.lfilter(self.db, self.da, LPR, zi=self.zi)

    # Ensure Bounds
    LPR = self.xp.clip(LPR, -1.0, 1.0)

    if self.cuda:
        return self.xp.asnumpy(LPR)
    
    return LPR