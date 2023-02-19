################################################################################
# Date: Nov-01-2022                                                            #
# Author(s): Yasir Ghunaim                                                     #
# E-mail: yasir.ghunaim@kaust.edu.sa                                           #
# Website: https://cemse.kaust.edu.sa/ivul                                     #
################################################################################

# A base class for training strategies in online streams.

from typing import Optional, List
from pkg_resources import parse_version

import torch
from torch.nn import Module, CrossEntropyLoss
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from avalanche.training.plugins.evaluation import default_evaluator
from avalanche.training.plugins import SupervisedPlugin, EvaluationPlugin
from avalanche.training.templates.supervised import SupervisedTemplate
from ...datasets.dataset_wrapper import OnlineDataset


class OnlineStream(SupervisedTemplate):
    """A base class for training strategies in online streams.

    This class adds the following changes to 'SupervisedTemplate':
    - Modifies the dataloader to return an online testing minibatch in addition to
      the training minibatch. The online testing minibatch is a copy of the training
      minibatch but with a 'test' augmentation. 
    - Defines below new properties of the minibatch: 
        1- test input
        2- test labels
        3- test index
        4- tets task_id
    """

    def __init__(
        self,
        model: Module,
        optimizer: Optimizer,
        criterion=CrossEntropyLoss(),
        train_mb_size: int = 1,
        eval_mb_size: int = None,
        device=None,
        plugins: Optional[List[SupervisedPlugin]] = None,
        evaluator: EvaluationPlugin = default_evaluator,
        eval_every=-1
    ):
        """Init.

        :param model: The model.
        :param optimizer: The optimizer to use.
        :param criterion: The loss criterion to use.
        :param train_mb_size: The train minibatch size. Defaults to 1.
        :param eval_mb_size: The eval minibatch size. Defaults to 1.
        :param device: The device to use. Defaults to None (cpu).
        :param plugins: Plugins to be added. Defaults to None.
        :param evaluator: (optional) instance of EvaluationPlugin for logging
            and metric computations. None to remove logging.
        :param eval_every: the frequency of the calls to `eval` inside the
            training loop. -1 disables the evaluation. 0 means `eval` is called
            only at the end of the learning experience. Values >0 mean that
            `eval` is called every `eval_every` epochs and at the end of the
            learning experience.
        """

        super().__init__(
            model,
            optimizer,
            criterion,
            train_epochs=1,
            train_mb_size=train_mb_size,
            eval_mb_size=eval_mb_size,
            device=device,
            plugins=plugins,
            evaluator=evaluator,
            eval_every=eval_every,
        )
        self.iteration_counter = 0

    def make_train_dataloader(
        self,
        num_workers=0,
        shuffle=False,
        pin_memory=True,
        persistent_workers=False,
        **kwargs
    ):
        """Data loader initialization.

        Called at the start of the online stream after the dataset adaptation.

        :param num_workers: number of thread workers for the data loading.
        :param shuffle: True if the data should be shuffled, False otherwise.
        :param pin_memory: If True, the data loader will copy Tensors into CUDA
            pinned memory before returning them. Defaults to True.
        """
        other_dataloader_args = {}

        if parse_version(torch.__version__) >= parse_version("1.7.0"):
            other_dataloader_args["persistent_workers"] = persistent_workers

        online_dataset = OnlineDataset(self.adapted_dataset)

        self.dataloader = DataLoader(
            online_dataset,
            num_workers=num_workers,
            batch_size=self.train_mb_size,
            shuffle=shuffle,
            pin_memory=pin_memory,
            **other_dataloader_args
        )

    @property
    def mb_index(self):
        """Current mini-batch indices."""
        return self.mbatch[2]

    @property
    def mb_test_x(self):
        """Current mini-batch test input."""
        return self.mbatch[3]

    @property
    def mb_test_y(self):
        """Current mini-batch test target."""
        return self.mbatch[4]

    @property
    def mb_test_index(self):
        """Current mini-batch test indices."""
        return self.mbatch[5]

    @property
    def mb_test_task_id(self):
        """Current mini-batch test task labels."""
        return self.mbatch[6]

    def training_epoch(self, **kwargs):
        """Training epoch.

        :param kwargs:
        :return:
        """
        for self.iteration_counter, self.mbatch in enumerate(self.dataloader):
            if self._stop_training:
                break

            self._unpack_minibatch()
            self._before_training_iteration(**kwargs)

            self.optimizer.zero_grad()
            self.loss = 0

            # Forward
            self._before_forward(**kwargs)
            self.mb_output = self.forward()
            self._after_forward(**kwargs)

            # Loss & Backward
            self.loss += self.criterion()

            self._before_backward(**kwargs)
            self.backward()
            self._after_backward(**kwargs)

            # Optimization step
            self._before_update(**kwargs)
            self.optimizer_step()
            self._after_update(**kwargs)

            self._after_training_iteration(**kwargs)