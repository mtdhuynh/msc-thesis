import random

import numpy as np
import torch

# Fix non-deterministic dataloading processes for reproducibility
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator().manual_seed(torch.initial_seed())

class ODDataloader(torch.utils.data.DataLoader):
    """
    Custom dataloader object for Object Detection (OD) tasks.
    Loads batches of data from the specified dataset object and performs shuffling.

    This class inherits everything from the torch.utils.data.DataLoaders class.
    It is used to add a layer of customizability in case it will be required in the future.
    """
    def __init__(self, dataset, batch_size, shuffle, num_workers, pin_memory):
        """
        Please, refer to the __init__ of torch.utils.data.DataLoaders.
        https://pytorch.org/docs/stable/_modules/torch/utils/data/dataloader.html#DataLoader
        """
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=lambda x: (torch.stack([xi for xi in zip(*x)][0]), [list(xi) for xi in zip(*x)][1]),
            worker_init_fn=seed_worker,
            generator=g
        )

        self.shuffle = shuffle

    def __repr__(self):
        return f'''ODDataloader(
            dataset={self.dataset}, 
            batch_size={self.batch_size}, 
            shuffle={self.shuffle}, 
            num_workers={self.num_workers}, 
            pin_memory={self.pin_memory}, 
            collate_fn={self.collate_fn}, 
            worker_init_fn={self.worker_init_fn},
            generator={self.generator}
        )'''