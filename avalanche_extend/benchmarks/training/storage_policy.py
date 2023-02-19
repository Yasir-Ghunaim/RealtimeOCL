################################################################################
# Date: Nov-01-2022                                                            #
# Author(s): Yasir Ghunaim                                                     #
# E-mail: yasir.ghunaim@kaust.edu.sa                                           #
# Website: https://cemse.kaust.edu.sa/ivul                                     #
################################################################################

# This is an modified version of reservoir sampling to work on a continuous 
# stream with no boundaries. The implementation is motivated by the following paper:
# Online Continual Learning with Maximal Interfered Retrieval
# https://arxiv.org/abs/1908.04742


from typing import TYPE_CHECKING
import torch
import numpy as np

if TYPE_CHECKING:
    from .templates.supervised import SupervisedTemplate


class OnlineReservoirSamplingBuffer():
    """Buffer updated with reservoir sampling."""
    # The algorithm follows
    # https://en.wikipedia.org/wiki/Reservoir_sampling
    # We sample a random uniform value in [0, 1] for each sample and
    # choose the `size` samples with higher values.
    # This is equivalent to a random selection of `size_samples`
    # from the entire stream.

    def __init__(self, mem_size: int, input_size, device):
        """
        :param mem_size: The total number of samples to be stored in the external memory.
        :param input_size: The shape of the input sample (e.g. [3, 256, 256])
        :param device: The device to use.
        """
        
        # INVARIANT: _buffer_weights is always sorted.
        self._buffer_weights = torch.zeros(0)
        self.mem_size = mem_size

        self.mem_x = torch.FloatTensor(mem_size, *input_size).fill_(0)
        self.mem_y = torch.LongTensor(mem_size).fill_(0)
        self.mem_index = torch.LongTensor(mem_size).fill_(0)
        self.mem_current_index = 0

        self.mem_x = self.mem_x.to(device)
        self.mem_y = self.mem_y.to(device)
        self.mem_index = self.mem_index.to(device)

        self.original_x = None

    def update(self, strategy: "SupervisedTemplate", **kwargs):
        """Update the buffer using the current minibatch.
        """

        new_x = self.original_x[:strategy.train_mb_size]
        new_y = strategy.mb_y[:strategy.train_mb_size]
        new_index = strategy.mb_index[:strategy.train_mb_size]
        new_weights = torch.rand(strategy.train_mb_size)

        cat_weights = torch.cat([new_weights, self._buffer_weights])
        sorted_weights, sorted_idxs = cat_weights.sort(descending=True)

        self.mem_current_index += strategy.train_mb_size
        self.mem_current_index = min(self.mem_size, self.mem_current_index)
        buffer_idxs = sorted_idxs[:self.mem_current_index]
        
        if self.mem_current_index == 0:
            self.mem_x = new_x[buffer_idxs]
            self.mem_y = new_y[buffer_idxs]
            self.mem_index = new_index[buffer_idxs]
        else:
            # Sort elements in the memory based on their weights
            select_ind = buffer_idxs - strategy.train_mb_size
            mem_valid_ind = select_ind[select_ind>=0]
            mem_target_ind = (select_ind>=0).nonzero(as_tuple=True)[0]


            # Since memory indices were sorted by weights in the previous iteration, we expect memory elements to maintain their rankings.
            # This means we simply need to shift elements to the right to allow the insertion of new elements coming from the current minibatch.
            # Ideally, we should be able to do this in one step. However, due to GPU limited memory, we chunk the operation in two steps.   
            mem_target_ind_rev = list(reversed(range(len(mem_target_ind))))
            chunks = np.array_split(mem_target_ind_rev, 2)
            for i in chunks:
                self.mem_x[mem_target_ind[i]] = self.mem_x[mem_valid_ind[i]]
                self.mem_y[mem_target_ind[i]] = self.mem_y[mem_valid_ind[i]]
                self.mem_index[mem_target_ind[i]] = self.mem_index[mem_valid_ind[i]]


            # Insert elements from current minibatch sorted based on their weights
            mb_valid_ind = buffer_idxs[buffer_idxs<strategy.train_mb_size]
            mb_target_ind = (buffer_idxs<strategy.train_mb_size).nonzero(as_tuple=True)[0]

            self.mem_x[mb_target_ind] = new_x[mb_valid_ind]
            self.mem_y[mb_target_ind] = new_y[mb_valid_ind]
            self.mem_index[mb_target_ind] = new_index[mb_valid_ind]
        
        self._buffer_weights = sorted_weights[: self.mem_current_index]

__all__ = [
    "OnlineReservoirSamplingBuffer"
]
