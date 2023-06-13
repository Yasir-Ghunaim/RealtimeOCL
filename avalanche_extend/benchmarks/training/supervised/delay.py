################################################################################
# Date: Nov-01-2022                                                            #
# Author(s): Yasir Ghunaim                                                     #
# E-mail: yasir.ghunaim@kaust.edu.sa                                           #
# Website: https://cemse.kaust.edu.sa/ivul                                     #
################################################################################

from typing import Optional, List

import torch
from torch.nn import Module, CrossEntropyLoss
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from avalanche.training.plugins.evaluation import default_evaluator
from avalanche.training.plugins import SupervisedPlugin, EvaluationPlugin
from .online_stream import OnlineStream

import copy

class Delay(OnlineStream):
    """Delay training strategy.
    Based on the determined delay (batch_delay argument), this strategy
    skips the appropriate number of mini-batches after each training iteration. 
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
        eval_every=-1,
        batch_delay=0,
        gradient_steps=1,
        output_dir=None,
        args = None
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
        :param batch_delay: the number of batches to skip after every training iteration.
        :param gradient_steps: the number of updates per training iteration.
        :param output_dir: Directory to save model checkpoints
        """

        super().__init__(
            model,
            optimizer,
            criterion,
            train_mb_size=train_mb_size,
            eval_mb_size=eval_mb_size,
            device=device,
            plugins=plugins,
            evaluator=evaluator,
            eval_every=eval_every,
        )
        
        self.batch_delay = batch_delay
        self.gradient_steps = gradient_steps
        self.train_mb_size = train_mb_size
        self.iteration_counter = 0
        self.training_counter = 0
        self.output_dir = output_dir
        self.args = args
        self.test_model = None

        assert self.output_dir != None, "output_dir must be defined."
        print("Initializing a Delay instance with batch_delay =", batch_delay, ", and gradient_steps =", self.gradient_steps)


    def is_training_batch(self):
        """Determine whether to train on a given minibatch
        """
        # Special case for LwF method
        if self.batch_delay == 0.33:
            return (self.iteration_counter % 4) < 3

        # Special case for MIR method
        elif self.batch_delay == 1.5:
            # Delay for 1 batch then delay for 2 batches
            return ((self.iteration_counter % 5 == 0) or ((self.iteration_counter-2) % 5 == 0))

        return self.iteration_counter % (int(self.batch_delay)+1) == 0

    def training_epoch(self, **kwargs):
        """Training epoch.
        """
        self.test_model = self.model

        self.one_third_of_stream = int(len(self.dataloader)/3)
        self.two_third_of_stream = int(2*len(self.dataloader)/3)

        for self.iteration_counter, self.mbatch in enumerate(self.dataloader): 
            if self._stop_training:
                break
            self._unpack_minibatch()

            # Copy model before training 
            if self.is_training_batch() and self.batch_delay > 0:
                self.test_model = copy.deepcopy(self.model)

            self._before_training_iteration(**kwargs)

            gradient_steps = int(self.gradient_steps)

            if self.is_training_batch():
                # Special case for LwF (fast-stream baseline)
                if self.batch_delay == 0.33 and self.gradient_steps == 1.33:
                    if self.iteration_counter % 4 == 0:
                        gradient_steps = 2
                    else:
                        gradient_steps = 1
                # Special case for LwF (slow-stream baseline)       
                elif self.batch_delay == 0 and self.gradient_steps == 1.33:
                    if self.iteration_counter % 3 == 0:
                        gradient_steps = 2
                    else:
                        gradient_steps = 1

                # Special case for MIR (fast-stream baseline)
                if self.batch_delay == 1.5 and self.gradient_steps == 2.5:
                    if self.iteration_counter % 5 == 0:
                        gradient_steps = 2
                    else:
                        gradient_steps = 3
                # Special case for MIR (slow-stream baseline)       
                elif self.batch_delay == 0 and self.gradient_steps == 2.5:
                    if self.iteration_counter % 2 == 0:
                        gradient_steps = 2
                    else:
                        gradient_steps = 3

                # if self.iteration_counter < 100:
                #     print("\nTRAIN:", "gradient_steps: ", gradient_steps)
                for self.step_count in range(gradient_steps):
                    self.optimizer.zero_grad()
                    self.loss = 0

                    # Forward
                    self._before_forward(**kwargs)
                    if self.iteration_counter == 0 or self.iteration_counter == 1 or self.iteration_counter == 4000:
                        print("\n========== (sanity check) shape of minibatch input x:", self.mb_x.shape)

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
            
            # Skip this minibatch
            else:
                if self.iteration_counter < 100:
                    print("DELAY")

                # The following logic is only required if you plan to use other
                # metrics from Avalanch that evalutes the model after each iteration.
                # For example, EpochAccuracy computes training accuracy after each iteration
                # based on the stored model predictions (self.mb_output). Since we are 
                # skipping this batch, we need to ensure that model predictions (self.mb_output)
                # are updated. Also, we want this (optional) evaluation to be performed on the 
                # test labels (self.mbatch[4]) and test task_id (self.mbatch[6]).

                self.test_model.eval()
                self.mb_output = self.test_model(self.mb_test_x)
                # Assign test labels to training labels 
                self.mbatch[1] = self.mbatch[4]
                # Assign test task_id to training task_id 
                self.mbatch[-1] = self.mbatch[6]
                self.test_model.train()

            self._after_training_iteration(**kwargs)

            if self.is_training_batch():
                self.training_counter += 1

            # Save model checkpoint at 1/3 and 2/3 of the stream for forward transfer
            if not self.args.debug:
                if self.iteration_counter == self.one_third_of_stream or self.iteration_counter == self.two_third_of_stream:
                    save_dict = {
                        'arch': self.args.arch,
                        'size_replay_buffer': self.args.size_replay_buffer,
                        'state_dict': self.model.state_dict(),
                        'optimizer' : self.optimizer.state_dict(),
                        'next_index': self.mb_index[:self.train_mb_size][-1] + 1
                    }
                    if self.iteration_counter == self.one_third_of_stream:
                        name = '/checkpoint_33_percent.pth.tar'
                    else:
                        name = '/checkpoint_67_percent.pth.tar'
                    print("Saving a checkpoint at:", self.output_dir + name)
                    torch.save(save_dict, self.output_dir + name)