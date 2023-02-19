################################################################################
# Date: Nov-01-2022                                                            #
# Author(s): Yasir Ghunaim                                                     #
# E-mail: yasir.ghunaim@kaust.edu.sa                                           #
# Website: https://cemse.kaust.edu.sa/ivul                                     #
################################################################################

from typing import Optional, TYPE_CHECKING
from pkg_resources import parse_version

from ...datasets.dataset_wrapper import OnlineDatasetWithReplay

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import Subset
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin
from torchvision import transforms
from random import choices

if TYPE_CHECKING:
    from avalanche.training.templates.supervised import SupervisedTemplate


class ReplayOnlinePlugin(SupervisedPlugin):
    """
    Experience replay plugin.
    Handles an external memory filled with randomly selected (sample with replacement)
    with patterns and implementing `before_training_exp` and `before_forward`
    callbacks.
    The `before_training_exp` callback is implemented to modifies the
    dataloder to include a FIFO memory buffer.

    The `before_forward` callback is implemented in order to concatenate training
    samples with buffer samples

    The :mem_size: attribute controls the total number of patterns to be stored
    in the external memory.
    """

    def __init__(
        self,
        mem_size: int = 200,
        gradient_steps: int = 1,
        batch_delay = 0,
        online_augmentation = transforms.Compose([
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip()]),
        seed = 0
    ):
        super().__init__()
        self.mem_size = mem_size
        self.gradient_steps = gradient_steps
        self.batch_delay = batch_delay
        self.seed = seed

        print("Replay online_augmentation:", online_augmentation)
        self.transform = online_augmentation

        self.original_x = None
        self.original_y = None
        self.original_index = None
        self.original_task_id = None
        torch.manual_seed(self.seed)

        print("Initializing a ReplayOnlinePlugin instance with mem_size:", mem_size, ", batch_delay:", batch_delay, ", gradient_steps:", gradient_steps)



    def before_training_exp(
        self,
        strategy: "OnlineStream",
        num_workers: int = 0,
        shuffle: bool = False,
        pin_memory=True,
        persistent_workers=False,
        **kwargs
    ):
        """
        Overwrites the dataloader to create a memory buffer and returns examples sampled
        from this buffer in each minibatch. 
        """

        other_dataloader_args = {}

        if parse_version(torch.__version__) >= parse_version("1.7.0"):
            other_dataloader_args["persistent_workers"] = persistent_workers

        datasetWithReplay = OnlineDatasetWithReplay(
            dataset=strategy.adapted_dataset,
            mem_size=self.mem_size,
            batch_size=strategy.train_mb_size,
            gradient_steps=self.gradient_steps,
            batch_delay=self.batch_delay,
            seed=self.seed)

        strategy.dataloader = DataLoader(
            datasetWithReplay,
            num_workers=num_workers,
            batch_size=strategy.train_mb_size,
            shuffle=shuffle,
            pin_memory=pin_memory,
            **other_dataloader_args
        )

    # Concatenate training samples with buffer samples
    def before_forward(self, strategy: "SupervisedTemplate", **kwargs):
        if len(strategy.mbatch) == 12:
            if self.gradient_steps == 1:
                strategy.mbatch[0] = torch.cat((strategy.mbatch[0], strategy.mbatch[7]), 0)
                strategy.mbatch[1] = torch.cat((strategy.mbatch[1], strategy.mbatch[8]), 0)
                strategy.mbatch[2] = torch.cat((strategy.mbatch[2], strategy.mbatch[9]), 0)
                strategy.mbatch[11] = torch.cat((strategy.mbatch[11], strategy.mbatch[10]), 0)
            # Since we are doing multiple gradient steps per training iteration,
            # we need to use different memory examples for each gradient step for better diversity.
            # For example, suppose gradient_steps=2 and train_mb_size=10 
            # (Note that memory batch size is equal to train_mb_size)
            # Then, memory inputs (strategy.mbatch[7]) will contains 20 samples
            # where the first 10 samples will be used for the first gradient step
            # and the second 10 samples will be used for the second gradient step.
            else:
                if strategy.step_count == 0:
                    # Store original training minibatch to be used for sebsequent concatenations
                    self.original_x = strategy.mbatch[0][:strategy.train_mb_size].clone()
                    self.original_y = strategy.mbatch[1][:strategy.train_mb_size].clone()
                    self.original_index = strategy.mbatch[2][:strategy.train_mb_size].clone()
                    self.original_task_id = strategy.mbatch[11][:strategy.train_mb_size].clone()
                strategy.mbatch[0] = torch.cat((self.original_x.clone(), strategy.mbatch[7][:,strategy.step_count]), 0)
                strategy.mbatch[1] = torch.cat((self.original_y, strategy.mbatch[8][:,strategy.step_count]), 0)
                strategy.mbatch[2] = torch.cat((self.original_index, strategy.mbatch[9][:,strategy.step_count]), 0)
                strategy.mbatch[11] = torch.cat((self.original_task_id, strategy.mbatch[10][:,strategy.step_count]), 0)
   
        strategy.mbatch[0] = self.transform(strategy.mbatch[0])

