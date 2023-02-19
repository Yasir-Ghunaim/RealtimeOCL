################################################################################
# Date: Nov-01-2022                                                            #
# Author(s): Yasir Ghunaim                                                     #
# E-mail: yasir.ghunaim@kaust.edu.sa                                           #
# Website: https://cemse.kaust.edu.sa/ivul                                     #
################################################################################

from torch.utils.data import Dataset
import torch
import math
import numpy as np

class IndexedDataset(Dataset):
    """ Modifies a generic dataset to return [sample, target, index] instead of only [sample, target].
        This is useful to make classic datasets (MNIST, CIFAR, etc.) compatible with the format used in our
        implementation of Online Continual Learning. """

    def __init__(self, dataset):
        """
        :param dataset (Dataset): an instance of Dataset.
        """
        self.dataset = dataset
        self.targets = [sample[1] for sample in self.dataset]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample, target = self.dataset[index]
        return sample, target, index

    
class OnlineDataset(Dataset):
    """
    Modifies a generic dataset to returns an online test minibatch in addition to the training minibatch.
    similar to the setup used in CLOC (https://arxiv.org/abs/2108.09020). Note that online test samples
    are copies of the training samples but differ in that they use 'test' augmentation. Online
    test samples are used to compute the average online accuracy.
    
    This class is used to prepare datasets in our setup before the training starts.
    """

    def __init__(self, dataset):
        """
        :param dataset (AvalancheDataset): an instance of AvalancheDataset that contains
            the training set.
        """
        self.train_set = dataset
        
        # Use evaluation transformation 
        self.online_test_set = dataset.eval()

    def __len__(self):
        return len(self.train_set)

    def __getitem__(self, index):
        sample, target, original_index, task_label = self.train_set[index]
        test_sample, test_target, test_original_index, test_task_label = self.online_test_set[index]
        return sample, target, original_index, \
            test_sample, test_target, test_original_index, test_task_label, task_label


class OnlineDatasetWithReplay(OnlineDataset):
    """Returns a training batch, an online test batch and buffer samples."""

    """
    Modifies a generic dataset to returns three mini-batches:
    1- Training mini-batch
    Used for training models. 

    2- Online test mini-batch
    Online test mini-batch are used to mimic the setup used in CLOC (https://arxiv.org/abs/2108.09020). Note that
    online test samples are copies of the training samples but differ in that they use 'test' augmentation. Online
    test samples are used to compute the average online accuracy.

    3- Buffer samples mini-batch
    These samples are concatenated with the training mini-batch. 

    """


    def __init__(self, dataset, mem_size, batch_size, gradient_steps=1, batch_delay=0, seed=0):
        """
        :param dataset (AvalancheDataset): an instance of AvalancheDataset that contains
            the training set.
        :param mem_size: Total number of patterns to be stored in the external memory.
        :param batch_size: mini-batch size for training.
        :param gradient_steps: the number of updates per training iteration.
        :param batch_delay: the number of batches to skip after every training iteration.
        """
        self.train_set = dataset
        # print("train_set transform:", self.train_set.transform)
        self.online_test_set = dataset.eval()
        # print("online_test_set transform:", self.online_test_set.transform)
        self.mem_size = mem_size
        self.batch_size = batch_size
        self.gradient_steps = gradient_steps
        self.batch_delay = batch_delay
        torch.manual_seed(seed)

        # =======================================
        # Pre-compute the sample indices that will be used in the training.
        # This is done by anticipating and removingindices that will be 
        # skipped by expensive methods.
        # =======================================
        self.training_index_list = torch.arange(0, len(dataset), dtype=torch.int64)
        self.mapped_index_list = torch.zeros(len(dataset), dtype=torch.int64)

        # Remove indices that are part of the delayed batches
        if batch_delay == 0.33:  # Special case for LwF method
            mask = (torch.floor(self.training_index_list/batch_size) % 4 == 3)
        elif batch_delay == 1.5: # Special case for MIR method
            mask1 = (torch.floor(self.training_index_list/batch_size) % 5) 
            mask2 = ((torch.floor(self.training_index_list/batch_size)-2) % 5)
        else:
            mask = (torch.floor(self.training_index_list/batch_size) % (int(batch_delay)+1))

        if batch_delay == 1.5: # Special case for MIR method
            first = self.training_index_list[mask1 == 0]
            second = self.training_index_list[mask2 == 0]
            self.training_index_list = torch.sort(torch.cat([first,second])).values
        else:
            self.training_index_list = self.training_index_list[mask == 0]

        
        # Given an index from the original dataset, we need a mapping to point to the correct location in
        # subset of training indices.
        self.mapped_index_list[self.training_index_list] = torch.arange(0, len(self.training_index_list), dtype=torch.int64)
    
    def is_training_batch(self, iteration_counter):
        """Determine whether the model will be training on the current minibatch or skipping it
        """
        # Special case for LwF method
        if self.batch_delay == 0.33:
            return (iteration_counter % 4) < 3
        # Special case for MIR method
        elif self.batch_delay == 1.5:
            # Delay for 1 batch then delay for 2 batches
            return ((iteration_counter % 5 == 0) or ((iteration_counter-2) % 5 == 0))

        return iteration_counter % (int(self.batch_delay)+1) == 0

    def __getitem__(self, index):
        sample, target, original_index, task_label = self.train_set[index]
        test_sample, test_target, test_original_index, test_task_label = self.online_test_set[index]
        
        iteration_counter = math.floor(index/self.batch_size)
        if iteration_counter > 0 and self.is_training_batch(iteration_counter):
            # Sample X indices from memory excluding skipped samples (due to delay)
            # where X = self.gradient_steps
            end_index = iteration_counter*self.batch_size
            mapped_end_index = self.mapped_index_list[end_index].item()
            buffer_index_list = self.training_index_list[max(0,mapped_end_index-self.mem_size):mapped_end_index]
            rand_ind = torch.randint(0, len(buffer_index_list), (self.gradient_steps,)).tolist()
            sampled_index = buffer_index_list[rand_ind].tolist()

            buffer_sample, buffer_target, buffer_original_index, buffer_task_label = self.train_set[sampled_index]
            
            return sample, target, original_index, \
                test_sample, test_target, test_original_index, test_task_label, \
                buffer_sample, buffer_target, buffer_original_index, buffer_task_label, \
                task_label

        return sample, target, original_index, \
            test_sample, test_target, test_original_index, test_task_label, task_label



__all__ = ["OnlineDataset", "OnlineDatasetWithReplay"]