# Guide to Measuring Computational Cost and Delay of Online Continual Learning (OCL) Methods

This guide provides instructions on how to measure the computational cost and associated delay of OCL methods using the [FlopsProfiler](https://www.deepspeed.ai/tutorials/flops-profiler/) tool from DeepSpeed version 0.6.7.
However, there are some limitations to using this tool that need to be considered:

1. FlopsProfiler does not work on methods that create deep copies of the original model, such as PoLRS, MIR, RWalk, and LwF.

2. FlopsProfiler only computes forward passes on the model, excluding the backward passes.

3. FlopsProfiler may not work when running on multiple GPUs, so it's essential to use a single GPU when profiling methods.

Given these limitations, we highly recommend that you manually calculate the forward/backward passes for each method first, and then use this tool as a verification step.

To address the first limitation, we define a new profiler instance to track each deepcopy instance of the model and create a counter to keep track of the FLOPs. However, when computing the FLOPs for the copied models, it is not possible to simultaneously compute the FLOPs for the original model. As a result, for methods that perform a deepcopy of the original model, only the additional FLOPs on top of the baseline ER can be measured.

To address the second limitation, a manual calculation is necessary to take into account the backward passes. The cost of a backward pass is two times the cost of a forward pass, as stated in the FlopsProfiler [documentation](https://www.deepspeed.ai/tutorials/flops-profiler/#flops-measurement). If the target OCL method performs additional backward passes, such as RWalk, the cost of forward passes needs to be multiplied by three.

Here is an example of measuring the computational cost of the baseline (ER) and one of the OCL methods (RWalk) on the CIFAR10 dataset:

## Computional Cost of the Baseline
To measure the computational cost of the baseline (ER), run the following script:
```
python main.py \
--dataset 'cifar10' \
--batch_size 10 \
--lr 0.05 \
--lr_type 'constant' \
--workers 6 \
--method 'ER' \
--seed 123 \
--debug \
--validation \
--profile_flops \
--size_replay_buffer 100
```

The output should include the following value:
```
fwd flops per GPU:                                            219.31 G
```

Since this only computes the forward passes, the cost of backward passes needs to be added to it. Given that the cost of a backward pass is two times the cost of a forward pass ([reference](https://www.deepspeed.ai/tutorials/flops-profiler/#flops-measurement)), the total computational cost of the ER baseline is:
```
219.31 G * 3 = 657.93 G
```
Therefore, **657.93 G** is the total computational cost of the ER baseline.


## Computational Cost of the Target Method
In this example, we use RWalk as the target method. A similar approach can be followed for other OCL methods.

As mentioned earlier, if an OCL method creates a deepcopy instance of the original model, FlopsProfiler will only monitor the original model but not the copied instance. To track new deepcopy instances of the model, the flag --profile_flops_deepcopy is used (works with MIR, LwF, and RWalk). Note that with this flag, the computational cost is measured only on the copied model so we need to manually add the baseline computational cost. 

To measure the computational cost of the RWalk method, run the following script:
```
python main.py \
--dataset 'cifar10' \
--batch_size 10 \
--lr 0.05 \
--lr_type 'constant' \
--workers 6 \
--method 'RWalk' \
--seed 123 \
--debug \
--validation \
--RWalk_update_freq 10 \
--RWalk_warmup 0 \
--profile_flops_deepcopy \
--size_replay_buffer 100
```

The output should include the following value:
```
The additional fwd flops used by this method is: 217.06 G
```

As mentioned earlier, this only takes into account the additional forward passes used by RWalk, excluding the main forward passes used for generating predictions.

To compute the computational cost, follow these steps:

1. Identify whether the method (RWalk) performs additional backward passes. For RWalk, the "loss.backward()" function can be found inside the "_update_grad" function in the following file: 
```
avalanche_extend/benchmarks/training/plugins/rwalk_online.py
```
This means RWalk performs additional backward passes that need to be added to the computational cost. 

2. Calculate the additional FLOPs used by RWalk by multiplying the cost of forward passes by three (as mentioned earlier).
```
Forward FLOPs = 217.06 G
backward FLOPs = Forward FLOPs * 2 =  217.06 G * 2 = 434.12 G
Total additional FLOPs by RWalk = 217.06 G + 434.12 G = 651.18 G
```


3. If the --profile_flops_deepcopy flag was used, the computational cost only takes into account the additional FLOPs of the method. This step is not necessary for methods that do not use deepcopy of the model. Therefore, to compute the total computational cost of the OCL method, the FLOPs of the original model used to generate the prediction (which is identical to the ER baseline) need to be added to the additional FLOPs of the method.
```
RWalk FLOPs = Baseline FLOPs + Additional FLOPs = 657.93 G + 651.18 G = 1309.11 G
```

4. Measure the complexity and delay of RWalk
```
RWalk FLOPs / Baseline FLOPs = 1309.11 G / 657.93 G = 2
RWalk Delay = complexity - 1 = 2 - 1 = 1
```

Therefore, the computational compliexty of the RWalk is **2**, and its delay is **1**.
