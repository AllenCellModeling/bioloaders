import numpy as np

import pdb

def index2onehot(indices, nlabels = None):

    if nlabels is None:
        nlabels = np.max(indices)
    
    ndat = len(indices)
    
    onehot = np.zeros([ndat, nlabels])
    onehot[np.arange(0, ndat), indices] = 1
    
    return onehot
    