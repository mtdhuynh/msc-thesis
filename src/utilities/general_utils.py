import os
import random

import numpy as np
import torch

def fix_seed(seed):
    """
    This function attempts to fix every possible source of randomness 
    due to non-deterministic behaviours of algorithms, pseudo-random generators, 
    other operations, etc., to increase reproducibility of results
    across different runs.

    As stated in the docs (https://pytorch.org/docs/stable/notes/randomness.html),
    deterministic behaviours may be slower than non-deterministic operations, and
    some operations do not have a deterministic counterpart.
    Therefore, fluctuations in performances across different experiments and runs
    can still be expected.

    Parameters:
        seed (int)  : a fixed seed.
    """
    ## Global
    os.environ['PYTHONHASHSEED'] = str(seed)

    ## Python
    random.seed(seed)

    ## Numpy
    np.random.seed(seed)

    ## PyTorch
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed) # All GPUs

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_timestamp(seconds):
    """
    Return string with time formatted in days, hours, minutes, seconds:
    '<dd>d <hh>h <mm>m <ss>s'

    N.B.: Code will break if time exceeds 31 days.

    Parameters:
        seconds (int)   : time to convert in [s].
    
    Returns:
        timestamp (str)
    """
    
    minutes, secs = divmod(int(seconds), 60)
    hours, mins = divmod(minutes, 60)
    days, hrs = divmod(hours, 24)
    
    return f'{days:d}d {hrs:02d}h {mins:02d}m {secs:02d}s'