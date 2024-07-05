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
from pkg_resources import parse_version


from ...datasets.dataset_wrapper import OnlineDataset

import torch
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin
from torch.utils.data import DataLoader
from torchvision import transforms

if TYPE_CHECKING:
    from avalanche.training.templates.supervised import SupervisedTemplate


class GSS_greedyOnlinePlugin(SupervisedPlugin):
    """
    Gradient based Sample Selection plugin,
    Implements the strategy defined in
    "Gradient based sample selection for online continual learning"
    https://arxiv.org/abs/1903.08671
    """

    def __init__(
        self,
        mem_size=200,
        mem_strength=10,
        input_size=[],
        threshold = 0.0,
        min_replacement=0,
        online_augmentation = transforms.Compose([
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip()]),
        seed = 0
    ):
        """
        :param mem_size: Total number of patterns to be stored in the external memory.
        :param mem_strength: Controls the number of times to compute gradients from memory.
        :param input_size: The shape of the input sample (e.g. [3, 256, 256]).
        :param threshold: (optional) This is used for debugging, it controls the
        similarity threshold to perform sample replacement.
        :param min_replacement: (optional) This is used for debugging, it enforces a minimum number 
        of samples to be replaced per training step. 
        :param online_augmentation: The augmentation to apply on the fly.
        """
        super().__init__()
        self.mem_size = mem_size
        self.mem_strength = mem_strength
        torch.manual_seed(seed)
        self.device = "cpu"
        self.threshold = threshold
        self.min_replacement = min_replacement

        self.ext_mem_list_x = torch.FloatTensor(mem_size, *input_size).fill_(0)
        self.ext_mem_list_y = torch.LongTensor(mem_size).fill_(0)
        self.ext_mem_list_index = torch.LongTensor(mem_size).fill_(0)
        self.ext_mem_list_current_index = 0
        self.replacement_counter = 0

        self.buffer_score = torch.FloatTensor(self.mem_size).fill_(0)

        print("GSS online_augmentation:", online_augmentation)
        self.transform = online_augmentation
        
        self.original_x = None

        print("Initializing an GSS_greedyOnlinePlugin instance with mem_size:", mem_size, " and mem_strength:", mem_strength)

    def before_training(self, strategy: "SupervisedTemplate", **kwargs):
        self.device = strategy.device
        self.ext_mem_list_x = self.ext_mem_list_x.to(strategy.device)
        self.ext_mem_list_y = self.ext_mem_list_y.to(strategy.device)
        self.ext_mem_list_index = self.ext_mem_list_index.to(strategy.device)
        self.buffer_score = self.buffer_score.to(strategy.device)

    def cosine_similarity(self, x1, x2=None, eps=1e-8):
        x2 = x1 if x2 is None else x2
        w1 = x1.norm(p=2, dim=1, keepdim=True)

        w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
        sim = torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)
        return sim

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

    def get_batch_sim(self, strategy, grad_dims, batch_x, batch_y):
        """
        Args:
            buffer: memory buffer
            grad_dims: gradient dimensions
            batch_x: current batch x
            batch_y: current batch y
        Returns: score of current batch, gradient from memory subsets
        """
        mem_grads = self.get_rand_mem_grads(strategy, grad_dims, len(batch_x))
        strategy.model.zero_grad()
        loss = strategy._criterion(strategy.model(batch_x), batch_y)
        loss.backward()
        batch_grad = self.get_grad_vector(
            strategy.model.parameters, grad_dims
        ).unsqueeze(0)
        batch_sim = max(self.cosine_similarity(mem_grads, batch_grad))
        return batch_sim, mem_grads

    def get_rand_mem_grads(self, strategy, grad_dims, gss_batch_size):
        """
        Args:
            buffer: memory buffer
            grad_dims: gradient dimensions
        Returns: gradient from memory subsets
        """
        temp_gss_batch_size = min(
            gss_batch_size, self.ext_mem_list_current_index
        )
        num_mem_subs = min(
            self.mem_strength, self.ext_mem_list_current_index // gss_batch_size
        )
        mem_grads = torch.zeros(
            num_mem_subs,
            sum(grad_dims),
            dtype=torch.float32,
            device=self.device,
        )
        shuffeled_inds = torch.randperm(
            self.ext_mem_list_current_index, device=self.device
        )
        for i in range(num_mem_subs):
            random_batch_inds = shuffeled_inds[
                i * temp_gss_batch_size : i * temp_gss_batch_size
                + temp_gss_batch_size
            ]
            batch_x = self.transform(self.ext_mem_list_x[random_batch_inds]).to(strategy.device)
            batch_y = self.ext_mem_list_y[random_batch_inds].to(strategy.device)

            strategy.model.zero_grad()

            loss = strategy._criterion(strategy.model(batch_x), batch_y)
            loss.backward()
            mem_grads[i].data.copy_(
                self.get_grad_vector(strategy.model.parameters, grad_dims)
            )
        return mem_grads


    def get_each_batch_sample_sim(
        self, strategy, grad_dims, mem_grads, batch_x, batch_y
    ):
        """
        Args:
            buffer: memory buffer
            grad_dims: gradient dimensions
            mem_grads: gradient from memory subsets
            batch_x: batch images
            batch_y: batch labels
        Returns: score of each sample from current batch
        """
        cosine_sim = torch.zeros(batch_x.size(0), device=strategy.device)
        for i, (x, y) in enumerate(zip(batch_x, batch_y)):
            strategy.model.zero_grad()
            ptloss = strategy._criterion(
                strategy.model(x.unsqueeze(0)), y.unsqueeze(0)
            )
            ptloss.backward()
            # add the new grad to the memory grads and add it is cosine
            # similarity
            this_grad = self.get_grad_vector(
                strategy.model.parameters, grad_dims
            ).unsqueeze(0)
            cosine_sim[i] = max(self.cosine_similarity(mem_grads, this_grad))

        return cosine_sim

    def before_training_exp(
        self, strategy, num_workers=0, shuffle: bool = False, pin_memory=True, persistent_workers=False, **kwargs
    ):
        """
        Overwrites the dataloader to return both train and online test samples. Note that online test samples
        are copies of the training samples but differ in that they use 'test' augmentation.
        """
        other_dataloader_args = {}
        if parse_version(torch.__version__) >= parse_version("1.7.0"):
            other_dataloader_args["persistent_workers"] = persistent_workers

        self.cloc_dataset = OnlineDataset(
            dataset=strategy.adapted_dataset
            )

        strategy.dataloader = DataLoader(
            self.cloc_dataset,
            num_workers=num_workers,
            batch_size=strategy.train_mb_size,
            shuffle=shuffle,
            pin_memory=pin_memory,
            **other_dataloader_args
        )

    def before_training_iteration(self, strategy: "OnlineStream", **kwargs):
        # TODO: Ensure that strategy is a Delay instance, otherwise remove the if condition 
        # This condition is used with Delay strategy 
        if strategy.iteration_counter % (strategy.batch_delay+1) == 0:
            self.original_x = strategy.mbatch[0].clone()
            if self.ext_mem_list_current_index > 0:
                # Random sample from replay buffer
                indices_to_sample = torch.randperm(self.ext_mem_list_current_index)[:strategy.train_mb_size].tolist()
                sampled_x = self.ext_mem_list_x[indices_to_sample]
                sampled_y = self.ext_mem_list_y[indices_to_sample]
                sampled_indices = self.ext_mem_list_index[indices_to_sample]
                
                strategy.mbatch[0] = self.transform(torch.cat((strategy.mbatch[0], sampled_x.to(strategy.device)), 0))
                strategy.mbatch[1] = torch.cat((strategy.mbatch[1], sampled_y.to(strategy.device)), 0)
                strategy.mbatch[2] = torch.cat((strategy.mbatch[2], sampled_indices.to(strategy.device)), 0)
                # Task ID should be 0 for all images
                strategy.mbatch[7] = torch.cat((strategy.mbatch[7], strategy.mbatch[7]), 0)
            else:
                strategy.mbatch[0] = self.transform(strategy.mbatch[0])
    def after_training_iteration(self, strategy, **kwargs):
        """
        After every training iteration this function select samples to fill
        the memory buffer based on cosine similarity
        """
        # TODO: Ensure that strategy is a Delay instance, otherwise remove the if condition 
        # This condition is used with Delay strategy 
        if strategy.iteration_counter % (strategy.batch_delay+1) == 0:
            strategy.model.eval()
            batch_size = strategy.train_mb_size

            # Compute the gradient dimension
            grad_dims = []
            for param in strategy.model.parameters():
                grad_dims.append(param.data.numel())

            place_left = (
                self.ext_mem_list_x.size(0) - self.ext_mem_list_current_index
            )
            if place_left <= 0:  # buffer full

                batch_sim, mem_grads = self.get_batch_sim(
                    strategy,
                    grad_dims,
                    batch_x=strategy.mb_x[:batch_size],
                    batch_y=strategy.mb_y[:batch_size],
                )

                # print("\nbatch_sim: %.3f" % batch_sim.item())
                if batch_sim < self.threshold:
                    # print("batch_sim < ", self.threshold)
                    buffer_score = self.buffer_score[
                        : self.ext_mem_list_current_index
                    ].cpu()

                    buffer_sim = (buffer_score - torch.min(buffer_score)) / (
                        (torch.max(buffer_score) - torch.min(buffer_score)) + 0.01
                    )

                    # draw candidates for replacement from the buffer
                    index = torch.multinomial(
                        buffer_sim, strategy.mb_x[:batch_size].size(0), replacement=False
                    ).to(strategy.device)

                    # estimate the similarity of each sample in the received batch
                    # to the randomly drawn samples from the buffer.
                    batch_item_sim = self.get_each_batch_sample_sim(
                        strategy, grad_dims, mem_grads, strategy.mb_x[:batch_size], strategy.mb_y[:batch_size]
                    )

                    # normalize to [0,1]
                    scaled_batch_item_sim = ((batch_item_sim + 1) / 2).unsqueeze(1)
                    buffer_repl_batch_sim = (
                        (self.buffer_score[index] + 1) / 2
                    ).unsqueeze(1)

                    # draw an event to decide on replacement decision
                    outcome = torch.multinomial(
                        torch.cat(
                            (scaled_batch_item_sim, buffer_repl_batch_sim), dim=1
                        ),
                        1,
                        replacement=False,
                    )
                    # replace samples with outcome =1
                    added_indx = torch.arange(
                        end=batch_item_sim.size(0), device=strategy.device
                    )
                    sub_index = outcome.squeeze(1).bool()

                    replacements = torch.sum(outcome).item()
                    self.replacement_counter += replacements
                    # print("Replacements: ", replacements)
                    # print("Total replacements: ", self.replacement_counter)
                    self.ext_mem_list_x[index[sub_index]] = self.original_x[:batch_size][
                        added_indx[sub_index]
                    ].clone()
                    self.ext_mem_list_y[index[sub_index]] = strategy.mb_y[:batch_size][
                        added_indx[sub_index]
                    ].clone()
                    self.ext_mem_list_index[index[sub_index]] = strategy.mb_index[:batch_size][
                        added_indx[sub_index]
                    ].clone()
                    self.buffer_score[index[sub_index]] = batch_item_sim[
                        added_indx[sub_index]
                    ].clone()
                elif self.min_replacement > 0:
                    # Enfore a minimum replacement
                    buffer_score = self.buffer_score[
                        : self.ext_mem_list_current_index
                    ].cpu()

                    buffer_sim = (buffer_score - torch.min(buffer_score)) / (
                        (torch.max(buffer_score) - torch.min(buffer_score)) + 0.01
                    )

                    # draw candidates for replacement from the buffer
                    index = torch.multinomial(
                        buffer_sim, strategy.mb_x[:batch_size].size(0), replacement=False
                    ).to(strategy.device)

                    # estimate the similarity of each sample in the received batch
                    # to the randomly drawn samples from the buffer.
                    batch_item_sim = self.get_each_batch_sample_sim(
                        strategy, grad_dims, mem_grads, strategy.mb_x[:batch_size], strategy.mb_y[:batch_size]
                    )

                    # normalize to [0,1]
                    scaled_batch_item_sim = ((batch_item_sim + 1) / 2).unsqueeze(1)
                    buffer_repl_batch_sim = (
                        (self.buffer_score[index] + 1) / 2
                    ).unsqueeze(1)

                    # Force replacement
                    numberOfReplacement = min(self.min_replacement, batch_item_sim.size(0))
                    lowK = torch.topk(scaled_batch_item_sim.squeeze(), numberOfReplacement, largest=False).indices
                    outcome = torch.zeros(batch_item_sim.size(0))
                    outcome[lowK] = 1
                    outcome = outcome.unsqueeze(1)

                    # replace samples with outcome =1
                    added_indx = torch.arange(
                        end=batch_item_sim.size(0), device=strategy.device
                    )
                    sub_index = outcome.squeeze(1).bool()

                    replacements = torch.sum(outcome).item()
                    self.replacement_counter += replacements
                    print("Replacements: ", replacements)
                    print("Total replacements: ", self.replacement_counter)
                    self.ext_mem_list_x[index[sub_index]] = self.original_x[:batch_size][
                        added_indx[sub_index]
                    ].clone()
                    self.ext_mem_list_y[index[sub_index]] = strategy.mb_y[:batch_size][
                        added_indx[sub_index]
                    ].clone()
                    self.ext_mem_list_index[index[sub_index]] = strategy.mb_index[:batch_size][
                        added_indx[sub_index]
                    ].clone()
                    self.buffer_score[index[sub_index]] = batch_item_sim[
                        added_indx[sub_index]
                    ].clone()

            else:
                offset = min(place_left, strategy.mb_x[:batch_size].size(0))
                updated_mb_x = strategy.mb_x[:batch_size][:offset]
                updated_mb_index = strategy.mb_index[:batch_size][:offset]
                updated_mb_y = strategy.mb_y[:batch_size][:offset]

                # first buffer insertion
                if self.ext_mem_list_current_index == 0:
                    batch_sample_memory_cos = (
                        torch.zeros(updated_mb_x.size(0)) + 0.1
                    )
                else:
                    # draw random samples from buffer
                    mem_grads = self.get_rand_mem_grads(
                        strategy=strategy,
                        grad_dims=grad_dims,
                        gss_batch_size=len(strategy.mb_x[:batch_size]),
                    )
                    # estimate a score for each added sample
                    batch_sample_memory_cos = self.get_each_batch_sample_sim(
                        strategy, grad_dims, mem_grads, updated_mb_x, updated_mb_y
                    )

                curr_idx = self.ext_mem_list_current_index
                self.ext_mem_list_x[curr_idx : curr_idx + offset].data.copy_(
                    self.original_x[:batch_size][:offset]
                )
                self.ext_mem_list_y[curr_idx : curr_idx + offset].data.copy_(
                    updated_mb_y
                )
                self.ext_mem_list_index[curr_idx : curr_idx + offset].data.copy_(
                    updated_mb_index
                )
                self.buffer_score[curr_idx : curr_idx + offset].data.copy_(
                    batch_sample_memory_cos
                )
                self.ext_mem_list_current_index += offset

            strategy.model.train()
