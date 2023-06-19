################################################################################
# Date: Nov-01-2022                                                            #
# Author(s): Yasir Ghunaim                                                     #
# E-mail: yasir.ghunaim@kaust.edu.sa                                           #
# Website: https://cemse.kaust.edu.sa/ivul                                     #
################################################################################

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
import torchvision.models as models
from torchvision import transforms

import os 
import argparse
import time
import math
import random
import numpy as np
from deepspeed.profiling.flops_profiler import FlopsProfiler

from avalanche_extend.benchmarks.classic import CLOC
from avalanche_extend.benchmarks.training.supervised import Delay, DelayPoLRS
from avalanche_extend.benchmarks.training.plugins import ReplayOnlinePlugin, \
    GSS_greedyOnlinePlugin, MIROnlinePlugin, \
        ER_ACE_OnlinePlugin, LwFOnlinePlugin, \
        RWalkOnlinePlugin
from avalanche_extend.evaluation.metrics import online_accuracy_metric, accuracy_metrics
from avalanche_extend.benchmarks.classic import SplitCIFAR10, SplitCIFAR100

from avalanche.training.plugins import EvaluationPlugin
from avalanche.evaluation.metrics import loss_metrics
from avalanche.logging import InteractiveLogger, TensorboardLogger, WandBLogger
from avalanche.models import SlimResNet18
from evaluate_additional_metrics import compute_backward_transfer, compute_forward_transfer

def main(args):
    print(args)
    print("======================")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(args.seed)

    # Flag to enable FLOPS profiler
    profiler_enabled = args.debug and (args.profile_flops or args.profile_flops_deepcopy)
    profiler_deepcopy_enabled = args.debug and args.profile_flops_deepcopy

    if profiler_deepcopy_enabled:
        print("Enabling FLOPs profling on a method with deepcopy")
    elif profiler_enabled:
        print("Enabling FLOPs profling")

    # Config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model
    if args.dataset == "cloc":
        n_classes = 713
        if args.pretrained:
            print("Using pre-trained model '{}'".format(args.arch))
            model = models.__dict__[args.arch](pretrained=True)
            model.fc = nn.Linear(2048, n_classes)
        else:
            print("Creating model '{}'".format(args.arch))
            model = models.__dict__[args.arch](num_classes = n_classes)

    elif args.dataset == "cifar10":
        model = SlimResNet18(10)

    elif args.dataset == "cifar100":
        model = SlimResNet18(100)


    # Define training augmentation for datasets
    CLOC_transform = transforms.Compose(
        [transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),])

    CIFAR_transform = transforms.Compose(
        [
            # transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
            ),
        ])


    # CL Benchmark Creation
    if args.dataset == "cloc":
        print("Creating CLOC benchmark")
        scenario = CLOC(n_experiences=1,
                        train_transform = CLOC_transform,
                        dataset_root=args.dataset_root,
                        debug=args.debug,
                        validation=args.validation)

        online_augmentation = transforms.Compose([
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip()])
        input_size=[3, 256, 256]

    elif args.dataset == "cifar10":            
        print("Creating CIFAR10 benchmark")
        scenario = SplitCIFAR10(
            n_experiences=1, return_task_id=False, seed=args.seed, train_transform=CIFAR_transform, 
            indexed=True, validation=args.validation
        )

        online_augmentation = transforms.RandomHorizontalFlip()
        input_size=[3, 32, 32]


    elif args.dataset == "cifar100":            
        print("Creating CIFAR100 benchmark")
        scenario = SplitCIFAR100(
            n_experiences=1, return_task_id=False, seed=args.seed, train_transform=CIFAR_transform, 
            indexed=True, validation=args.validation
        )

        online_augmentation = transforms.RandomHorizontalFlip()
        input_size=[3, 32, 32]

    
    # Prepare for training & testing
    model = torch.nn.DataParallel(model)
    optimizer = SGD(model.parameters(), lr=args.lr,
                    momentum=args.momentum,
                    weight_decay=args.weight_decay)

    criterion = CrossEntropyLoss()


    # DEFINE THE EVALUATION PLUGIN
    # The evaluation plugin manages the metrics computation.
    # It takes as argument a list of metrics, collectes their results and returns
    # them to the strategy it is attached to.
    loggers = []

    suffix = 'Delay'
    suffix += '_{}'.format(args.dataset)
    suffix += '_bs{}'.format(args.batch_size)
    suffix += '_lr{}'.format(args.lr)
    suffix += '_delay{}'.format(args.batch_delay)
    suffix += '_steps{}'.format(args.gradient_steps)
    suffix += '_buffer{}'.format(args.size_replay_buffer)
    suffix += '_{}'.format(args.method)
    suffix += '_{}'.format(args.lr_type)
    if args.pretrained:
        suffix += '_pretrained'
    
    output_directory = args.output_dir + "/" + suffix
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    if args.wandb:
        wandbLogger = WandBLogger(
            project_name="real_time_ocl",
            run_name=suffix,
            config={
                "batch_size": args.batch_size,
                "lr": args.lr,
                "delay": args.batch_delay,
                "steps": args.gradient_steps,
                "buffer": args.size_replay_buffer,
                "method": args.method,
                "lr_type": args.lr_type,
                "pretrained": args.pretrained,
                "comment": "REAL_TIME_OCL_REPRODUCE",
            },
            # params={"tags:": [args.method, args.lr_type, args.arch, args.pretrained]},
            dir=output_directory,
        )
        loggers.append(wandbLogger)


    tensorboardLogger = None
    if not args.debug and args.lr_type != "polrs":
        # log to Tensorboard
        tensorboardLogger = TensorboardLogger(tb_log_dir=output_directory)
        loggers.append(tensorboardLogger)

    # print to stdout
    loggers.append(InteractiveLogger())

    if args.lr_type == "polrs" or profiler_enabled:
        # Disable Avalanche metrics
        # For PoLRS, we follow the implementaiton of the original paper which did not utilize Avalanche (refer to DelayPoLRS)
        # For the profiler, we disable the metrics, as this could result in inflated FLOPs values
        eval_plugin = EvaluationPlugin(
            loggers=loggers,
            benchmark=scenario,
            strict_checks=False
        )
    else:
        eval_plugin = EvaluationPlugin(
            accuracy_metrics(epoch=True, experience=True),
            loss_metrics(epoch_running=True),
            online_accuracy_metric(),
            loggers=loggers,
            benchmark=scenario,
            strict_checks=False
        )

    # Continual learning strategies
    plugins = None

    print(f"Using the CL method: {args.method}")
    if args.method == "ER" and args.size_replay_buffer > 0:
        plugins = [ReplayOnlinePlugin(mem_size=args.size_replay_buffer, gradient_steps=math.ceil(args.gradient_steps), 
         batch_delay=args.batch_delay, online_augmentation=online_augmentation, seed=args.seed)]

    elif args.method == "GSS" and args.size_replay_buffer > 0:
        plugins = [GSS_greedyOnlinePlugin(mem_size=args.size_replay_buffer, mem_strength=args.GSS_mem_strength,
         input_size=input_size, threshold=args.GSS_threshold,
         min_replacement= args.GSS_min_replacement, online_augmentation=online_augmentation, seed=args.seed)]
    
    elif args.method == "MIR" and args.size_replay_buffer > 0:
        plugins = [MIROnlinePlugin(mem_size=args.size_replay_buffer, batch_size_mem=args.batch_size,
        input_size=input_size, device=device, online_augmentation=online_augmentation,
        profile=profiler_deepcopy_enabled, seed=args.seed)]

    elif args.method == "ACE" and args.size_replay_buffer > 0:
        plugins = [ER_ACE_OnlinePlugin(mem_size=args.size_replay_buffer,
        device=device, online_augmentation=online_augmentation, seed=args.seed)]

    elif args.method == "LwF" and args.size_replay_buffer > 0:
        plugins = [ReplayOnlinePlugin(mem_size=args.size_replay_buffer, gradient_steps=math.ceil(args.gradient_steps), 
         batch_delay=args.batch_delay, online_augmentation=online_augmentation, seed=args.seed),
                    LwFOnlinePlugin(warmup=args.LwF_warmup, update_freq=args.LwF_update_freq,
                    profile=profiler_deepcopy_enabled, seed=args.seed)]

    elif args.method == "RWalk" and args.size_replay_buffer > 0:
        plugins = [ReplayOnlinePlugin(mem_size=args.size_replay_buffer, gradient_steps=math.ceil(args.gradient_steps), 
         batch_delay=args.batch_delay, online_augmentation=online_augmentation, seed=args.seed),
                    RWalkOnlinePlugin(ewc_lambda=args.RWalk_ewc_lambda, warmup=args.RWalk_warmup,
                    update_freq=args.RWalk_update_freq, profile=profiler_deepcopy_enabled, seed=args.seed)]


    if args.lr_type == "polrs":
        cl_strategy = DelayPoLRS(
            model,
            optimizer,
            criterion,
            train_mb_size=args.batch_size,
            eval_mb_size=args.batch_size,
            device=device,
            evaluator=eval_plugin,
            batch_delay=args.batch_delay,
            output_dir = output_directory,
            args = args,
            plugins=plugins
        )
    else:
        cl_strategy = Delay(
            model,
            optimizer,
            criterion,
            train_mb_size=args.batch_size,
            eval_mb_size=args.batch_size,
            device=device,
            evaluator=eval_plugin,
            batch_delay=args.batch_delay,
            plugins=plugins,
            gradient_steps=args.gradient_steps,
            output_dir = output_directory,
            args = args
        )

    # ======== Train and Evaluation ========
    results = []
    print("Starting experiment...")

    train_stream = scenario.train_stream
    test_stream = scenario.test_stream

    # Start computing the FLOPS used by the selected CL method
    # When profiling methods that deepcopy the model, it is necessary to disable this profiler to prevent interference with the method's internal profiler
    if profiler_enabled and not args.profile_flops_deepcopy:
        prof = FlopsProfiler(model)
        prof.start_profile()
    
    start_time = time.time()
    # Since we don't use tasks boundaries, the entire stream is loaded to a single experience
    experience = train_stream[0]

    print("Number of samples in the training stream: ", len(experience.dataset))
    # Train on the stream
    cl_strategy.train(experience, shuffle=False, num_workers=args.workers)

    # Evaluate on the held-out test set at the end of training
    # Note, during validation (hyperparameter search) or profiling, we don't need to evaluate on the test set
    if not (args.validation or profiler_enabled):
        results.append(cl_strategy.eval(test_stream, num_workers=args.workers))
   
    print("--- Training completed in %s seconds ---" % (time.time() - start_time))

    if profiler_enabled and not args.profile_flops_deepcopy:
        prof.stop_profile()
        prof.print_model_profile(detailed=False)
        prof.end_profile()

    # Save final model
    if not args.debug:
        if args.lr_type == "polrs":
            for i in range(args.NOModels):
                save_dict = {
                    'arch': args.arch,
                    'size_replay_buffer': args.size_replay_buffer,
                    'state_dict': cl_strategy.model_pool[i].state_dict(),
                    'optimizer' : cl_strategy.optim_pool[i].state_dict()
                }
                torch.save(save_dict, output_directory + '/final_model' + str(i) + '.pth.tar')
        else:   
            save_dict = {
                'arch': args.arch,
                'size_replay_buffer': args.size_replay_buffer,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict()
            }
            torch.save(save_dict, output_directory + '/final_model.pth.tar')

   
    # This code computes the backward/forward transfer metric.
    # The code was written for an ablation study in the appendix section of our paper.
    # It only works with the CLOC dataset.
    if args.dataset == "cloc" and not args.debug and not profiler_enabled:

        if args.lr_type == "polrs":
            # Define Tensorboard for PoLRS, which defines tensorboard logging inside DelayPoLRS 
            tensorboardLogger = TensorboardLogger(tb_log_dir=output_directory)

        compute_backward_transfer(args, model, scenario, device, tensorboardLogger)

        checkpoint_path = output_directory + '/checkpoint_33_percent.pth.tar'
        compute_forward_transfer(args, model, scenario, device, tensorboardLogger, checkpoint_path, 33)

        checkpoint_path = output_directory + '/checkpoint_67_percent.pth.tar'
        compute_forward_transfer(args, model, scenario, device, tensorboardLogger, checkpoint_path, 67)


if __name__ == "__main__":

    model_names = sorted(name for name in models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(models.__dict__[name]))

    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
					choices=model_names,
					help='model architecture: ' +
						' | '.join(model_names))
    parser.add_argument('--dataset', default='cloc',
                    choices=['cloc', 'cifar10', 'cifar100'],
					help='Lists of datasets')
    parser.add_argument('--dataset_root', default="path/to/cloc_dataset/release/", 
					help='The path to the release folder of CLOC dataset')               
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
					help='use pre-trained model')
    parser.add_argument('--validation', action='store_true',
					help='use validation set')


    parser.add_argument('--method', default="None", 
					help='The CL method to use')
    parser.add_argument('--size_replay_buffer', default=0, 
					type=int, help='size of the experience replay buffer')
    parser.add_argument('--gradient_steps', default=1, 
					type=float, help='number of gradient steps per minibatch')
    parser.add_argument('--batch_size', default=128, type=int, metavar='N',
					help='mini-batch size for training and testing dataloaders')

    parser.add_argument('--workers', default=10, type=int, metavar='N',
					help='number of data loading workers')
    parser.add_argument('--lr_type',  default="constant",
                    help='The learning rate schedule method')
    parser.add_argument('--lr',  default=0.05, type=float,
					metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
					help='momentum')
    parser.add_argument('--wd', '--weight_decay', default=1e-4, type=float,
					metavar='W', help='weight decay (default: 1e-4)', dest='weight_decay')
    parser.add_argument('--batch_delay', default=0, type=float,
					help='the amount of batch delay')



    parser.add_argument('--debug', dest='debug', action='store_true',
					help='debugging disable tensorboard logging')
    parser.add_argument('--profile_flops', action='store_true',
					help='estimate the flops for training')
    parser.add_argument('--profile_flops_deepcopy', action='store_true',
					help='estimate the flops for methods that use deepcopy of the main model')
    parser.add_argument('--seed', default=0, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--output_dir', default='./results',
                        help='Directory to store Tensorboard output.')

    # LwF
    parser.add_argument('--LwF_warmup', default=0.05, 
					type=float, help='percentage of the data stream to train on before enabling LwF')
    parser.add_argument('--LwF_update_freq', default=1000, 
					type=int, help='the frequency (in terms of number of iterations) to update LwF previous model')

    # RWalk
    parser.add_argument('--RWalk_warmup', default=0.05, 
					type=float, help='percentage of the data stream to train on before enabling RWalk')
    parser.add_argument('--RWalk_update_freq', default=1000, 
					type=int, help='the frequency (in terms of number of iterations) to update RWalk')
    parser.add_argument('--RWalk_ewc_lambda', default=1, 
					type=float, help='hyperparameter to weigh the penalty inside the RWalk loss')

    # PoLRS
    parser.add_argument("--NOModels", default = 3, type = int, help = 'how many models to train simultaneously')
    parser.add_argument("--LRMultiFactor", default = 2.0, type = float, help = 'factor to change LR between models')
    parser.add_argument("--LR_adjust_intv", default = 5, type = int, help = 'adjust LR in how many epochs')

    # GSS
    parser.add_argument('--GSS_mem_strength', default=10, 
					type=int, help='GSS memory strength')
    parser.add_argument('--GSS_threshold', default=0.0, 
					type=float, help='GSS similarity threshold')
    parser.add_argument('--wandb', dest='wandb', action='store_true', 
                    help='use wandb for logging')
    parser.add_argument('--GSS_min_replacement', default=0, 
					type=int, help='GSS enforce minimum replacement')

    args = parser.parse_args()

    assert args.lr_type in ["constant", "polrs"]

    main(args)
