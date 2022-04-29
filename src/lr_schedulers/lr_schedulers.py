import torch

class Constant_LR(torch.optim.lr_scheduler._LRScheduler):
    """
    Constant learning rate "scheduler". 
    The default scheduler if none is specified in the configuration file. 
    """
    def __init__(self, optimizer, last_epoch=-1):
        """
        Creates a _LRScheduler object with default settings from the optimizer passed as input.

        Parameters:
            optimizer (torch.optim) : optimization algorithm specified.
            last_epoch (int)        : last epoch index.
        """
        super().__init__(optimizer, last_epoch)

        self.optimizer = optimizer
        self.last_epoch = last_epoch

    def __repr__(self):
        return f'Constant_LR(optimizer={self.optimizer}, last_epoch={self.last_epoch})'

    def get_lr(self):
        return [base_lr for base_lr in self.base_lrs]

class LR_Scheduler():
    """
    Learning rate scheduler from torch.optim.lr_scheduler with custom parameters. 
    """
    def __init__(self, optimizer, name, last_epoch=-1, **params):
        """
        Loads a learning rate scheduler for the specified optimization algorithm.

        Parameters: 
            optimizer (torch.optim) : optimization algorithm specified.
            last_epoch (int)        : last epoch index.
            **params (dict)         : admissible torch.optim.lr_scheduler kwargs (learning rate configuration parameters - initial/last lr, decay, etc.).
        """
        self.optimizer = optimizer
        self.lr_name = name
        self.params = params
        self.last_epoch = last_epoch

        try:
            scheduler_func = torch.optim.lr_scheduler.__dict__[self.lr_name]
        except Exception as e:
            print(e)
            print(f'Unknown scheduler name: {self.lr_name}. Available learning rate schedulers: {torch.optim.lr_scheduler.__dict__.keys()}')

        self.lr_scheduler = scheduler_func(self.optimizer, last_epoch=last_epoch, **params)