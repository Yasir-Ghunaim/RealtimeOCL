################################################################################
# Date: Nov-01-2022                                                            #
# Author(s): Yasir Ghunaim                                                     #
# E-mail: yasir.ghunaim@kaust.edu.sa                                           #
# Website: https://cemse.kaust.edu.sa/ivul                                     #
################################################################################
"""
Code adapted from the Avalanche repository:
https://github.com/ContinualAI/avalanche/blob/master/avalanche/training/plugins/lwf.py
- Mainly, we removed the dependency on task boundaries (experiencies) to 
enable working with a continuous stream.
"""


import copy
import torch

from avalanche.models import avalanche_forward
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin
from deepspeed.profiling.flops_profiler import FlopsProfiler


class LwFOnlinePlugin(SupervisedPlugin):
    """
    A Learning without Forgetting plugin.
    LwF uses distillation to regularize the current loss with soft targets
    taken from a previous version of the model.
    """

    def __init__(self, alpha=1, temperature=2, warmup=0.05, update_freq=1000, profile = False, seed = 0):
        """
        :param alpha: distillation hyperparameter.
        :param temperature: softmax temperature for distillation
        :param warmup: percentage of the data stream to train on before enabling LwF. We introduce this 
                       in the online implementation as the model initally will not be suitable for distillation.
        :param update_freq: the frequency (in terms of number of iterations) to update the previous model
                            and seen classes.
        :param profile: compute the additional flops required by this method
        """

        super().__init__()

        self.alpha = alpha
        self.temperature = temperature
        self.warmup = warmup
        self.update_freq = update_freq
        self.prev_model = None
        self.seed = seed
        self.just_finished_warmup = True
        self.profile_enabled = profile

        self.flops_counter = 0
        self.prev_classes = set()
        self.seen_classes = set()
        """ In Avalanche, targets of different experiences are not ordered. 
        As a result, some units may be allocated even though their 
        corresponding class has never been seen by the model.
        Knowledge distillation uses only units corresponding to old classes. 
        """

        torch.manual_seed(self.seed)
        print("Initializing an LwFOnlinePlugin instance with warmup:", self.warmup, " and update_freq:", self.update_freq)

    def _distillation_loss(self, out, prev_out, active_units):
        """
        Compute distillation loss between output of the current model and
        and output of the previous (saved) model.
        """
        # we compute the loss only on the previously active units.
        au = list(active_units)

        log_p = torch.log_softmax(out[:, au] / self.temperature, dim=1)
        q = torch.softmax(prev_out[:, au] / self.temperature, dim=1)
        res = torch.nn.functional.kl_div(log_p, q, reduction="batchmean")
        return res

    def penalty(self, out, x, alpha, curr_model):
        """
        Compute weighted distillation loss.
        """

        if self.prev_model is None:
            return 0
        else:
            with torch.no_grad():
                y_prev = self.prev_model(x)
                y_curr = out

            dist_loss = 0
            # compute kd only for previously seen classes.
            yp = y_prev
            yc = y_curr
            au = self.prev_classes
            dist_loss += self._distillation_loss(yc, yp, au)
            return alpha * dist_loss

    def before_backward(self, strategy, **kwargs):
        """
        Add distillation loss
        """
        penalty = self.penalty(
            strategy.mb_output, strategy.mb_x, self.alpha, strategy.model
        )
        strategy.loss += penalty

    # def after_training_exp(self, strategy, **kwargs):
    def after_training_iteration(self, strategy, **kwargs):
        """
        Save a copy of the model after 'update_freq' iterations and
        update self.prev_classes to include the previously seen classes.
        """
        if strategy.is_training_batch():
            batch_size = strategy.train_mb_size
            current_classes = set(strategy.mb_y[:batch_size].tolist())
            self.seen_classes = self.seen_classes.union(current_classes)

            # Enable LwF after passing through "warmup" iterations of the datastream
            if (strategy.iteration_counter / len(strategy.dataloader)) > self.warmup:
                # Save copy of model and seen classes every 'update_freq' iterations
                if self.just_finished_warmup or ((strategy.training_counter + 1) % self.update_freq == 0):
                    self.just_finished_warmup = False

                    if self.profile_enabled and self.prev_model is not None:
                        self.flops_counter += self.prof.get_total_flops()
                        self.prof.stop_profile()
                        self.prof.end_profile()
                        
                    self.prev_model = copy.deepcopy(strategy.model)
                    self.prev_classes = copy.deepcopy(self.seen_classes)
                    print("Saving new model, with num of classes: ", len(self.prev_classes))

                    if self.profile_enabled:
                        self.prof = FlopsProfiler(self.prev_model)
                        self.prof.start_profile()

    def after_training_epoch(self, strategy, **kwargs):
        if self.profile_enabled:
            # This calculation excludes the FLOPs required to perform forward passes on the data (i.e., ER baseline FLOPs)  
            print("The additional fwd flops used by this method is: {:.2f} G".format(self.flops_counter/(10**9)))
