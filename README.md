
# Real-Time Evaluation in Online Continual Learning: A New Hope (CVPR2023, Highlight)
A realistic and fair evaluation for Online Continual Learning (OCL), in which the computational cost of OCL methods is incorporated into the training process. That is, more expensive OCL methods than our baseline ends up training on proportionally fewer data.

#### This work was accepted at CVPR 2023 as a highlight (top 2.5%) paper. [Link to preprint](https://arxiv.org/abs/2302.01047)

## Requirements
Install the conda enviornment by running: 
```bash
    conda create -n realtime_ocl python=3.9
    conda activate realtime_ocl
    pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
```

## Usage
This repo is based on the Avalanche framework. We highly recommend going over [this quick tutorial](https://avalanche.continualai.org/from-zero-to-hero-tutorial/01_introduction) to understand the basics of Avalanche.

To reproduce the experiments in the paper, please perform the following steps:

**Step 1: Install the CLOC dataset**

To install the CLOC dataset, clone the dataset repository from GitHub and follow the instructions provided in the README file, which can be found at the following [link](https://github.com/IntelLabs/continuallearning/tree/main/CLOC).

**Step 2: Preprocess CLOC**

When downloading CLOC images through Step 1, the script will access a large list of image URLs hosted on Flickr server. However, due to potential issues such as changes in the availability of the images, it is possible that some images will be missing after the download process is complete.

To ensure optimal training time and minimize errors during dataloading, we strongly recommend removing these invalid pointers from the CLOC metadata by running the "[preprocess_CLOC.ipynb](https://github.com/Yasir-Ghunaim/RealtimeOCL/blob/main/preprocess_CLOC.ipynb)" notebook provided in our repo.

**Step 3: Download CLOC cross validation set**

The default CLOC metadata only contains train and test splits. However, if you want to access the cross-validation set that we used, you need to download the metadata files starting with "cross_val_" from the following [link](https://drive.google.com/file/d/1pu8fHz9uCp8z5Ec6sScO3L4y943OZ-il/view?usp=drivesdk).

**Step 4: Run the experiments**

To reproduce the results presented in Figures 3-5 for our paper, please run the required scripts:
-   For Fast Stream experiments (Figure 3), check out the following [scripts](https://github.com/Yasir-Ghunaim/RealtimeOCL/tree/main/experiments/CLOC/fast_stream).  
    
-   For Fast Stream with Data Normalization (Figure 4), check out the following [scripts](https://github.com/Yasir-Ghunaim/RealtimeOCL/tree/main/experiments/CLOC/fast_stream_norm). ER-- curves can be obtained from these scripts, while ER and other OCL methods curves can be obtained from the Fast Stream scripts mentioned above.

-   For Slow stream experiments (Figure 5), check out the following [scripts](https://github.com/Yasir-Ghunaim/RealtimeOCL/tree/main/experiments/CLOC/slow_stream). ER curve can simply be obtained from the fast stream experiments.

## Project Structure

```bash
├── avalanche_extend                    # Avalanche Framework Extensions
│ ├── benchmarks							
│ │ ├── classic                         # Benchmark generator for datasets
│ │ ├── datasets                        # Custom dataset definitions 
│ │ ├── scenarios                       # Helpers for benchmark generator  
│ │ └── training                        # Training pipeline and OCL strategies
│ │     ├── plugins                     # OCL strategies
│ │     └── supervised                  # Contains our proposed fast stream setup (called Delay)
│ └── evaluation                        # Metrics (e.g., Average Online Accuracy)
├── experiments                         # Experiments scripts (in SLURM format, but they can also be used as bash scripts.)
│ ├── CIFAR10
│ │   ├── fast_stream
│ │   └── slow_stream
│ ├── CIFAR100
│ │   ├── fast_stream
│ │   └── slow_stream
│ └── CLOC
│     ├── fast_stream
│     ├── fast_stream_norm
│     └── slow_stream
├── main.py                             # Instantiate training pipeline, OCL strategies, metrics and loggers 
├── evaluate_additional_metrics.py      # Evaluates traditional CL metrics (i.e., Backward/Forward Transfer)
```

## Measuring Training Complexity and Delay of OCL Methods
To measure the training complexity of OCL methods, we first manually calculated the number of forward/backward passes for each method. Then, we used the [FlopsProfiler](https://www.deepspeed.ai/tutorials/flops-profiler/) tool for verification.

To use FlopsProfiler, please follow our tutorial, which can be found at the following [link](https://github.com/Yasir-Ghunaim/RealtimeOCL/blob/main/measuring_delay.md). 

Please keep in mind that the FlopsProfiler tool has some limitations, as outlined in the tutorial. Thus, it is strongly recommended that you manually calculate the forward/backward passes for each method and then use this tool for verification purposes.

## Cite
```
@inproceedings{ghunaim2023real,
  title={Real-Time Evaluation in Online Continual Learning: A New Hope},
  author={Ghunaim, Yasir and Bibi, Adel and Alhamoud, Kumail and Alfarra, Motasem and Hammoud, Hasan Abed Al Kader and Prabhu, Ameya and Torr, Philip HS and Ghanem, Bernard},
  booktitle = {Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2023}
}
```
