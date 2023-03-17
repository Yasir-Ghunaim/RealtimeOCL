################################################################################
# Date: Nov-01-2022                                                            #
# Author(s): Yasir Ghunaim                                                     #
# E-mail: yasir.ghunaim@kaust.edu.sa                                           #
# Website: https://cemse.kaust.edu.sa/ivul                                     #
################################################################################
"""
Code adapted from the Avalanche repository:
https://github.com/ContinualAI/avalanche/blob/master/avalanche/training/plugins/gss_greedy.py
- Mainly, we removed the dependency on task boundaries (experiencies) to 
enable working with a continuous stream.
"""

from typing import TYPE_CHECKING

from avalanche.training.plugins.strategy_plugin import SupervisedPlugin
from ..storage_policy import OnlineReservoirSamplingBuffer
from torchvision import transforms

import torch
import torch.nn.functional as F
import copy

if TYPE_CHECKING:
    from avalanche.training.templates.supervised import SupervisedTemplate

class MIROnlinePlugin(SupervisedPlugin):
    """
    Maximally Interfered Retrieval plugin,
    Implements the strategy defined in
    "Online Continual Learning with Maximally Interfered Retrieval"
    https://arxiv.org/abs/1908.04742

    Handles an external memory filled with reservoir sampling
    and implementing `before_training_iteration` and `after_training_iteration`
    callbacks.

    The `before_training_iteration` callback is implemented in order to use the
    dataloader that creates mini-batches with examples from both training
    data and external memory.

    The `after_training_iteration` callback is implemented in order to add new
    patterns to the external memory.

    :param mem_size: The total number of patterns to be stored in the external memory.
    :param batch_size_mem: The number of memory samples to return. In our setup, this parameter
    is set to equal the training minibatch size. 
    :param subsample_size: The size of the subset that is sampled from the memory.  
    :param input_size: The shape of the input sample (e.g. [3, 256, 256]).
    :param device: The device to use.
    :param online_augmentation: The augmentation to apply on the fly.
    """

    def __init__(
        self,
        mem_size: int = 200,
        batch_size_mem: int = None,
        subsample_size: int = None,
        input_size = [],
        device = "cpu",
        online_augmentation = transforms.Compose([
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip()]),
        seed = 0
    ):
        super().__init__()
        assert batch_size_mem is not None, "batch_size_mem arg is required"

        self.mem_size = mem_size
        self.batch_size_mem = batch_size_mem
        if subsample_size is None:
            size_multiplier = 5 # This factor is chosen to match the repository of the MIR paper  
            subsample_size = batch_size_mem * size_multiplier
        self.subsample_size = subsample_size
        self.device = device
        self.flops = 0
        
        self.storage_policy = OnlineReservoirSamplingBuffer(mem_size=mem_size, input_size=input_size, device=device)
        torch.manual_seed(seed)

        print("MIR online_augmentation:", online_augmentation)
        self.transform = online_augmentation

        print("Initializing an MIROnlinePlugin instance with mem_size:", mem_size, " and subsample_size:", subsample_size)

    def get_grad_vector(self, pp, grad_dims):
        """
        gather the gradients in one vector
        """
        grads = torch.zeros(sum(grad_dims), device=self.device)
        grads.fill_(0.0)
        cnt = 0
        for param in pp():
            if param.grad is not None:
                beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
                en = sum(grad_dims[: cnt + 1])
                grads[beg:en].copy_(param.grad.data.view(-1))
            cnt += 1
        return grads

    def overwrite_grad(self, pp, new_grad, grad_dims):
        """
            This is used to overwrite the gradients with a new gradient
            vector, whenever violations occur.
            pp: parameters
            newgrad: corrected gradient
            grad_dims: list storing number of parameters at each layer
        """
        cnt = 0
        for param in pp():
            param.grad = torch.zeros_like(param.data)
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            this_grad = new_grad[beg: en].contiguous().view(
                param.data.size())
            param.grad.data.copy_(this_grad)
            cnt += 1

    def get_future_step_parameters(self, this_net, grad_vector, grad_dims, lr=1):
        """
        computes \theta-\delta\theta
        :param this_net:
        :param grad_vector:
        :return:
        """
        new_net = copy.deepcopy(this_net)
        self.overwrite_grad(new_net.parameters, grad_vector, grad_dims)
        with torch.no_grad():
            for param in new_net.parameters():
                if param.grad is not None:
                    param.data=param.data - lr*param.grad.data
        return new_net

    def before_training_iteration(self, strategy, **kwargs):
        if self.storage_policy.mem_current_index == 0:
            self.storage_policy.original_x = strategy.mbatch[0][:strategy.train_mb_size].clone()
            strategy.mbatch[0] = self.transform(strategy.mbatch[0])
            return

        # TODO: Ensure that strategy is a Delay instance, otherwise remove the if condition 
        # This condition is used with Delay strategy 
        if strategy.is_training_batch():
            indices_to_sample = torch.randperm(self.storage_policy.mem_current_index)[:self.subsample_size].tolist()
            
            buffer_x = self.transform(self.storage_policy.mem_x[indices_to_sample])            
            buffer_y = self.storage_policy.mem_y[indices_to_sample]
            buffer_ind = self.storage_policy.mem_index[indices_to_sample]

            grad_dims = []
            for param in strategy.model.parameters():
                grad_dims.append(param.data.numel())
            grad_vector = self.get_grad_vector(strategy.model.parameters, grad_dims)
            
            lr = 0.05
            for param_group in strategy.optimizer.param_groups:
                lr = param_group['lr']

            future_model = self.get_future_step_parameters(strategy.model, grad_vector, grad_dims, lr=lr)

            with torch.no_grad():
                logits_track_pre = strategy.model(buffer_x)
                logits_track_post = future_model(buffer_x)

                pre_loss = F.cross_entropy(logits_track_pre, buffer_y , reduction="none")
                # pre_loss =strategy._criterion(logits_buffer, mem_y)
                post_loss = F.cross_entropy(logits_track_post, buffer_y , reduction="none")
                scores = post_loss - pre_loss

                all_logits = scores
                big_ind = all_logits.sort(descending=True)[1][:self.batch_size_mem]
            
            mem_x, mem_y, mem_ind = buffer_x[big_ind], buffer_y[big_ind], buffer_ind[big_ind].to(self.device)

            self.storage_policy.original_x = strategy.mbatch[0][:strategy.train_mb_size].clone()
            strategy.mbatch[0] = self.transform(strategy.mbatch[0])
            
            strategy.mbatch[0] = torch.cat((strategy.mbatch[0], mem_x), 0)
            strategy.mbatch[1] = torch.cat((strategy.mbatch[1], mem_y), 0)
            strategy.mbatch[2] = torch.cat((strategy.mbatch[2], mem_ind), 0)
            # Task ID should be 0 for all images
            strategy.mbatch[7] = torch.cat((strategy.mbatch[7], strategy.mbatch[7]), 0)

    def after_training_iteration(self, strategy, **kwargs):
        # TODO: Ensure that strategy is a Delay instance, otherwise remove the if condition 
        # This condition is used with Delay strategy 
        if strategy.is_training_batch():
            self.storage_policy.update(strategy, **kwargs)
