import torch

class ODOptimizer():
    """
    Optimizer class for loading optimization algorithms, mostly based on torch.optim.Optimizer optimizers.
    """
    def __init__(self, model, name='Adam', **params):
        """
        Loads a torch.optim optimizer from the library. 

        Parameters:
            model (torch.nn.Module) : the provided object detection model.
            name (str)              : name of optimization algorithm from torch.optim.
            **params (dict)         : admissible torch.optim kwargs (optimizer configuration parameters - lr, weight decay, etc.).
        """
        self.model_params = model.parameters()
        self.optim_name = name
        self.params = params

        try:
            optimizer_func = torch.optim.__dict__[self.optim_name]
        except Exception as e:
            print(e)
            print(f'Unknown optimizer: {self.optim_name}. Available optimizers: {torch.optim.__dict__.keys()}.')
        
        self.optimizer = optimizer_func(self.model_params, **self.params)