################################################################################
# Date: Nov-01-2022                                                            #
# Author(s): Yasir Ghunaim                                                     #
# E-mail: yasir.ghunaim@kaust.edu.sa                                           #
# Website: https://cemse.kaust.edu.sa/ivul                                     #
################################################################################

from typing import Optional, TYPE_CHECKING
from pkg_resources import parse_version

from ...datasets.dataset_wrapper import OnlineDatasetWithReplay

from torch.utils.data import DataLoader, Dataset
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin
from avalanche.models import avalanche_forward
from torchvision import transforms

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from avalanche.training.templates.supervised import SupervisedTemplate


class ER_ACE_OnlinePlugin(SupervisedPlugin):
    """
    ER-ACE loss plugin.

    This code is adapted from the original implementation of ER-ACE to enable working with Avalanche.
    Original code: https://github.com/pclucas14/AML
    Paper: New Insights on Reducing Drastic Representation Drift in Online Continual Learning
    https://arxiv.org/abs/2104.05025
    """

    def __init__(
        self,
        mem_size: int = 200,
        online_augmentation = transforms.Compose([
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip()]),
        device = "cpu",
        seed = 0
    ):
        """
        :param mem_size: The total number of samples to be stored in the external memory.
        :param online_augmentation: The augmentation to apply on the fly.
        :param device: The device to use.
        """
        super().__init__()
        self.mem_size = mem_size
        self.device = device
        self.seed = seed

        self.seen_so_far = torch.LongTensor(size=(0,)).to(self.device)

        print("ACE online_augmentation:", online_augmentation)
        self.transform = online_augmentation
        
        torch.manual_seed(self.seed)
        print("Initializing an ER_ACE_OnlinePlugin instance")


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
            seed=self.seed)

        strategy.dataloader = DataLoader(
            datasetWithReplay,
            num_workers=num_workers,
            batch_size=strategy.train_mb_size,
            shuffle=shuffle,
            pin_memory=pin_memory,
            **other_dataloader_args
        )

    def before_forward(self, strategy: "SupervisedTemplate", **kwargs):
        x = strategy.mbatch[0]
        y = strategy.mbatch[1]

        if len(strategy.mbatch) == 12:
            # Forward on incoming data
            self.calculate_ace_loss(strategy, self.transform(x), y)

            # Prepare forward on replay data (the acutal forward pass happens in the strategy training_epoch())
            strategy.mbatch[0] = self.transform(strategy.mbatch[7])
            strategy.mbatch[1] = strategy.mbatch[8]
            strategy.mbatch[2] = strategy.mbatch[9]
            strategy.mbatch[11] = strategy.mbatch[10]
        else:
            # Skip ACE for the first iteration
            self.seen_so_far = torch.cat([self.seen_so_far, y.unique()]).unique()
            strategy.mbatch[0] = self.transform(strategy.mbatch[0])


    def calculate_ace_loss(self, strategy, x, y):
        present = y.unique()
        self.seen_so_far = torch.cat([self.seen_so_far, present]).unique()
        # print("self.seen_so_far:", self.seen_so_far)
        # process data
        logits = avalanche_forward(strategy.model, x, None)
        mask = torch.zeros_like(logits)

        # unmask current classes
        mask[:, present] = 1

        # unmask unseen classes
        unseen_classes = torch.range(0, logits.size(1)-1, dtype=torch.int64)
        m = torch.ones(unseen_classes.numel(), dtype=torch.bool)
        m[self.seen_so_far] = False
        unseen_classes = unseen_classes[m]
        mask[:, unseen_classes] = 1
        # mask[:, self.seen_so_far.max():] = 1

        logits  = logits.masked_fill(mask == 0, -1e9)

        strategy.loss = F.cross_entropy(logits, y)

