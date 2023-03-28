################################################################################
# Date: Nov-01-2022                                                            #
# Author(s): Yasir Ghunaim                                                     #
# E-mail: yasir.ghunaim@kaust.edu.sa                                           #
# Website: https://cemse.kaust.edu.sa/ivul                                     #
################################################################################

from typing import Optional, List

from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from avalanche.training.plugins.evaluation import default_evaluator
from avalanche.training.plugins import SupervisedPlugin, EvaluationPlugin
from .online_stream import OnlineStream

import math
import time
import torch
import os 
from tensorboardX import SummaryWriter
import copy

from deepspeed.profiling.flops_profiler import FlopsProfiler
import time

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def set(self, meter):
        self.val = meter.val
        self.avg = meter.avg
        self.sum = meter.sum
        self.count = meter.count

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)



class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        # for test only
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


class DelayPoLRS(OnlineStream):
    """Delay training strategy (special implementation for PoLRS).
    Based on the determined delay (batch_delay argument), this strategy
    skips the appropriate number of mini-batches after each training iteration. 

    Since implementing PoLRS Avalanche is a bit tricky, we simply copied
    the original PoLRS implementation define in this repo:
    https://github.com/IntelLabs/continuallearning/tree/main/CLOC
    which is based on the paper:
    Online Continual Learning with Natural Distribution Shifts: An Empirical Study with Visual Data
    https://arxiv.org/pdf/2108.09020.pdf
    Due to this design choice, the metrics are logged differently than other methods. 
    """

    def __init__(
        self,
        model: Module,
        optimizer: Optimizer,
        criterion,
        train_mb_size: int = 1,
        eval_mb_size: int = None,
        device=None,
        plugins: Optional[List[SupervisedPlugin]] = None,
        evaluator: EvaluationPlugin = default_evaluator,
        eval_every=-1,
        batch_delay=0,
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
            and metric computations.
        :param eval_every: the frequency of the calls to `eval` inside the
            training loop. -1 disables the evaluation. 0 means `eval` is called
            only at the end of the learning experience. Values >0 mean that
            `eval` is called every `eval_every` epochs and at the end of the
            learning experience.
        :param batch_delay: the number of batches to skip after every training iteration.
        :param output_dir: Directory to store Tensorboard output
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
        self.output_dir = output_dir

        NOModels = 3

        self.model_pool = [None] * NOModels
        self.optim_pool = [None] * NOModels
        LR_min = args.lr * args.LRMultiFactor
        self.meter_pool = [None] * NOModels
        self.writer_pool = [None] * NOModels

        for i in range(0, NOModels):
            if NOModels == 1 or i == 1:
                self.model_pool[i] = model
                self.optim_pool[i] = optimizer
            else:
                self.model_pool[i] = copy.deepcopy(model).cuda()
                self.optim_pool[i] = torch.optim.SGD(self.model_pool[i].parameters(), LR_min/float(args.LRMultiFactor**i),
                                        momentum=args.momentum,
                                        weight_decay=args.weight_decay)

            # init metrics for online fit
            self.meter_pool[i] = self.init_meters_for_hypTune()
            out_dir_curr = self.output_dir + "/model_{}".format(i)
            os.makedirs(out_dir_curr, exist_ok = True)
            self.writer_pool[i] = SummaryWriter(out_dir_curr)

            print("[model pool init] LR[{}] = {}".format(i, self.get_lr(self.optim_pool[i])))


        self.online_fit_meters = self.init_online_fit_meters()
        self.args = args
        self.mCriterion = criterion

        out_folder_eval = self.output_dir
        os.makedirs(out_folder_eval, exist_ok = True)
        self.writer = SummaryWriter(out_folder_eval)
        self.test_model = None

        print("Initializing a Delay instance with batch_delay =", batch_delay)

    def init_online_fit_meters(self):
        return [AverageMeter('LossOF', ':.4e'), AverageMeter('AccOF@1', ':6.2f'), AverageMeter('AccOF@5', ':6.2f')]

    def init_meters_for_hypTune(self):
        return [AverageMeter('Loss', ':.4e'), AverageMeter('AccF@1', ':6.2f'), AverageMeter('AccF@5', ':6.2f')]

    def init_local_meters(self):
        meter_local = {}
        meter_local['batch_time'] = AverageMeter('Time', ':6.3f')
        meter_local['data_time'] = AverageMeter('Data', ':6.3f')

        meter_local['losses'] = AverageMeter('Loss', ':.4e')
        
        meter_local['top1'] = AverageMeter('Acc@1', ':6.2f')
        meter_local['top5'] = AverageMeter('Acc@5', ':6.2f')
        
        meter_local['top1_future'] = AverageMeter('AccF@1', ':6.2f')
        meter_local['top5_future'] = AverageMeter('AccF@5', ':6.2f')

        meter_local['top1_Rep'] = AverageMeter('Acc_old@1', ':6.2f')
        meter_local['top5_Rep'] = AverageMeter('Acc_old@5', ':6.2f')

        return meter_local
        
    def write_tensor_board(self, meter_pool, meter_local, writer, writer_pool, online_fit_meters, iter_curr, extra_name = ''):
        if extra_name == '':
            name_epoch = '_epoch'
        else:
            name_epoch = ''

        writer.add_scalar("train_loss{}".format(extra_name), meter_local['losses'].avg, iter_curr)
        writer.add_scalar("train_acc1{}".format(extra_name), meter_local['top1'].avg, iter_curr)
        writer.add_scalar("train_acc5{}".format(extra_name), meter_local['top5'].avg, iter_curr)

        writer.add_scalar("avg_online_loss{}".format(extra_name+name_epoch), online_fit_meters[0].avg, iter_curr)
        writer.add_scalar("avg_online_acc1{}".format(extra_name+name_epoch), online_fit_meters[1].avg, iter_curr)
        writer.add_scalar("avg_online_acc5{}".format(extra_name+name_epoch), online_fit_meters[2].avg, iter_curr)

        # writer.add_scalar("avg_online_loss_time{}".format(extra_name), online_fit_meters[0].avg, time_last)
        # writer.add_scalar("avg_online_acc1_time{}".format(extra_name), online_fit_meters[1].avg, time_last)
        # writer.add_scalar("avg_online_acc5_time{}".format(extra_name), online_fit_meters[2].avg, time_last)

        writer.add_scalar("train_acc1_old{}".format(extra_name), meter_local['top1_Rep'].avg, iter_curr)
        writer.add_scalar("train_acc5_old{}".format(extra_name), meter_local['top5_Rep'].avg, iter_curr)
                    
        writer.add_scalar("train_acc1_future{}".format(extra_name), meter_local['top1_future'].avg, iter_curr)
        writer.add_scalar("train_acc5_future{}".format(extra_name), meter_local['top5_future'].avg, iter_curr)
        
        for i in range(len(meter_pool)):
            writer_pool[i].add_scalar("train_loss_future{}".format(extra_name), meter_pool[i][0].avg, iter_curr)
            writer_pool[i].add_scalar("train_acc1_future{}".format(extra_name), meter_pool[i][1].avg, iter_curr)
            writer_pool[i].add_scalar("train_acc5_future{}".format(extra_name), meter_pool[i][2].avg, iter_curr)

    def set_train(self, model_pool):
        for model in model_pool:
            model.train()

    def set_zero_grad(self, optim_pool):
        for optim in optim_pool:
            optim.zero_grad()

    def set_eval(self, model_pool):
        for model in model_pool:
            model.eval()

    def get_lr(self, optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']
    
    def reset_meter_pool(self, meter_pool):
        for i in range(len(meter_pool)):
            meter_pool[i].reset()

    def cloc_accuracy(self, output, target, topk=(1,), cross_GPU = False):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            if cross_GPU:
                output = output
                target = target
                
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.reshape(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res

    def population_based_LR_adjust(self, model_pool, optim_pool, meter_pool, idx_best_model, args):
        metric_middle = meter_pool[1][1]
        metric_best = meter_pool[idx_best_model][1]

        if metric_best == metric_middle:
            idx_best_model = 1

        LR_min = args.LRMultiFactor * self.get_lr(optim_pool[idx_best_model])

        for i in range(len(model_pool)):
            # self.flopsCounter[i] += self.prof[i].get_total_flops()
            # self.prof[i].stop_profile()
            # self.prof[i].end_profile()

            if i != idx_best_model:
                model_pool[i] = copy.deepcopy(model_pool[idx_best_model])

            # self.prof[i] = FlopsProfiler(model_pool[i])
            # self.prof[i].start_profile()

            print("[reset meter pool]: meter_pool[{}][0] = {}; idx_best_model = {}".format(i, meter_pool[i][0].avg, idx_best_model))
            self.reset_meter_pool(meter_pool[i])
            optim_pool[i] = torch.optim.SGD(model_pool[i].parameters(), LR_min /float(args.LRMultiFactor**i),
                                        momentum=args.momentum,
                                        weight_decay=args.weight_decay)

        print("[model copy]: setting lr[{}] to {}; meter_pool[i] = {}".format(i, self.get_lr(optim_pool[i]), meter_pool[i]))

    def test_on_model_pool(self, args, model_pool, images, target, meter_pool, online_fit_meters, meter_local, idx_best_model, criterion):
        with torch.no_grad():
            self.set_eval(model_pool) 
            output_pool = [None] * len(model_pool)
            
            # don't put anything inside this for loop to avoid synchronization			
            for i in range(len(model_pool)):
                output_pool[i] = model_pool[i](images)
            
            target_new = target.cuda()

            # compute online fit
            loss_best = 1e8
            idx_loss_best = 0
            loss_BCE_best = 1e8
            idx_loss_BCE_best = 0
            acc_best = -1.0
            idx_acc_best = 0
            acc5_best = -1.0
            idx_acc5_best = 0
            for i in range(len(model_pool)):
                output_all = output_pool[i].cuda()
                acc1F, acc5F = self.cloc_accuracy(output_all, target_new, topk=(1, 5))

                meter_pool[i][1].update(acc1F[0])
                if acc_best < meter_pool[i][1].avg:
                    idx_acc_best = i
                    acc_best = meter_pool[i][1].avg

                if i == idx_best_model:
                    meter_local['top1_future'].update(acc1F[0])
                    meter_local['top5_future'].update(acc5F[0])

                    # if idx_set is not None:
                    output_album = output_all
                    target_album = target_new
                    acc1OF, acc5OF = self.cloc_accuracy(output_album, target_album, topk=(1,5))
                    online_fit_meters[1].update(acc1OF[0])
                    online_fit_meters[2].update(acc5OF[0])
                        

            self.set_train(model_pool)

        return idx_acc_best


    def training_epoch(self, **kwargs):
        """Training epoch.

        :param kwargs:
        :return:
        """

        self.one_third_of_stream = int(len(self.dataloader)/3)
        self.two_third_of_stream = int(2*len(self.dataloader)/3)

        epoch_split = math.ceil(len(self.dataloader)/90)
        cloc_epoch_counter = 0
        meter_local = self.init_local_meters()

        progress = ProgressMeter(len(self.dataloader),
                                [meter_local['batch_time'], meter_local['data_time'], meter_local['losses'],
                                meter_local['top1'], meter_local['top1_future'], meter_local['top1_Rep'],
                                self.online_fit_meters[1], self.online_fit_meters[2]])
                                # prefix="Epoch: [{}]".format("!!!!")) 

        self.set_train(self.model_pool)
        end = time.time()

        self.set_zero_grad(self.optim_pool)
        idx_best_model = 1

        self.test_model = self.model_pool

        # for i in range(len(self.model_pool)):
        #     self.prof[i].start_profile()
        # start_time = time.time()

        for self.iteration_counter, self.mbatch in enumerate(self.dataloader):
            cloc_epoch_counter = math.floor(self.iteration_counter / epoch_split)

            if self._stop_training:
                break
            meter_local['batch_time'].update(time.time() - end)
            self._unpack_minibatch()

            # Copy model before training 
            if self.iteration_counter % (self.batch_delay+1) == 0 and self.batch_delay > 0:
                self.test_model = copy.deepcopy(self.model_pool)

            idx_best_model = self.test_on_model_pool(self.args, self.test_model,
                            self.mb_test_x, self.mb_test_y, self.meter_pool, self.online_fit_meters,
                            meter_local, idx_best_model, self.criterion)

            self.writer.add_scalar("idx_best_model", idx_best_model, cloc_epoch_counter*len(self.dataloader)+self.iteration_counter)

            self._before_training_iteration(**kwargs)

            if self.iteration_counter % (self.batch_delay+1) == 0:
                for i in range(len(self.model_pool)):			
                    self.optim_pool[i].zero_grad()   # reset gradient

                output = [None] * len(self.model_pool)
                loss_pool = [None] * len(self.model_pool)

                # Forward
                self._before_forward(**kwargs)
                if self.iteration_counter == 0 or self.iteration_counter == 1 or self.iteration_counter == 4000:
                        print("========== (sanity check) self.mb_x.shape:", self.mb_x.shape)
                for i in range(len(self.model_pool)):
                    output[i] = self.model_pool[i](self.mb_x)
                
                self.mb_output = output[idx_best_model]
                self._after_forward(**kwargs)

                # Loss & Backward
                for i in range(len(self.model_pool)):
                    loss_pool[i] = self.mCriterion(output[i], self.mb_y)

                self._before_backward(**kwargs)
                # self.backward()
                for i in range(len(self.model_pool)):			
                    loss_pool[i].backward()
                self._after_backward(**kwargs)

                # Optimization step
                self._before_update(**kwargs)
                for i in range(len(self.model_pool)):			
                    self.optim_pool[i].step()  
                self._after_update(**kwargs)
            else:
                # The following logic is only required if you plan to use other
                # metrics from Avalanch that evalutes the model after each iteration.
                # For example, EpochAccuracy computes training accuracy after each iteration
                # based on the stored model predictions (self.mb_output). Since we are 
                # skipping this batch, we need to ensure that model predictions (self.mb_output)
                # are updated. Also, we want this (optional) evaluation to be performed on the 
                # test labels (self.mbatch[4]) and test task_id (self.mbatch[6]).
                self.set_eval(self.test_model) 
                for i in range(len(self.model_pool)):
                    output[i] = self.test_model[i](self.mb_test_x)
                self.mb_output = output[idx_best_model]
                # Assign test labels to training labels 
                self.mbatch[1] = self.mbatch[4]
                # Assign test task_id to training task_id 
                self.mbatch[-1] = self.mbatch[6]
                self.set_train(self.test_model)


            meter_local['losses'].update(loss_pool[idx_best_model].item(), self.mb_y.numel())
            acc1, acc5 = self.cloc_accuracy(output[idx_best_model], self.mb_y, topk=(1, 5))
            meter_local['top1'].update(acc1[0], self.mb_y.numel())
            meter_local['top5'].update(acc5[0], self.mb_y.numel())
            meter_local['batch_time'].update(time.time() - end)
            end = time.time()

            if self.iteration_counter % (10) == 0:
                progress.display(self.iteration_counter)

                self.write_tensor_board(self.meter_pool, meter_local, self.writer, self.writer_pool, self.online_fit_meters, 
                    self.iteration_counter, extra_name = '_iter')


            if self.iteration_counter % epoch_split == 0:
                best_lr = self.get_lr(self.optim_pool[idx_best_model])
                self.writer.add_scalar("learning rate", best_lr, cloc_epoch_counter)

                if cloc_epoch_counter % self.args.LR_adjust_intv == 0:
                    print("[population_based_LR_adjust]: adjusting optimum lr...epoch = {}".format(cloc_epoch_counter))
                    print("idx_best_model: ", idx_best_model)
                    self.population_based_LR_adjust(self.model_pool, self.optim_pool, self.meter_pool, idx_best_model, self.args)


            self._after_training_iteration(**kwargs)

            # Save model checkpoint at 1/3 and 2/3 of the stream for forward transfer
            if self.iteration_counter == self.one_third_of_stream or self.iteration_counter == self.two_third_of_stream:
                for i in range(self.args.NOModels):
                    save_dict = {
                        'arch': self.args.arch,
                        'size_replay_buffer': self.args.size_replay_buffer,
                        'state_dict': self.model_pool[i].state_dict(),
                        'optimizer' : self.optim_pool[i].state_dict(),
                        'next_index': self.mb_index[:self.train_mb_size][-1] + 1,
                        'idx_best_model': idx_best_model
                    }

                    if self.iteration_counter == self.one_third_of_stream:
                        name = '/checkpoint_33_percent' + str(i) + '.pth.tar'
                    else:
                        name = '/checkpoint_67_percent' + str(i) + '.pth.tar'
                    print("Saving a checkpoint at:", self.args.output_dir + name)
                    torch.save(save_dict, self.args.output_dir + name)


        print("Training Loss =",  meter_local['losses'].avg)
        print("Top1_Average_Online_Accuracy =", self.online_fit_meters[1].avg.item())

        # print("--- %s seconds ---" % (time.time() - start_time))
        # for i in range(len(self.model_pool)):
        #     self.flopsCounter[i] += self.prof[i].get_total_flops()
        #     self.prof[i].stop_profile()
        #     self.prof[i].end_profile()
        #     print("Model ", i ," flops: ", self.flopsCounter[i])
        # write_tensor_board(self.meter_pool, meter_local, self.writer, self.writer_pool, 
        # self.online_fit_meters, kwargs["exp_counter"])
