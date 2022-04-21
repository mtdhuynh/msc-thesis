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


