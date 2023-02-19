# Online Continual Learning

## Requirements
Install the conda enviornment by running: 
```bash
    conda create -n realtime_ocl python=3.9
    conda activate realtime_ocl
    pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
    pip install avalanche-lib==0.2.0 deepspeed==0.6.7 tensorboardx
```

## Dataset Installation
We use CLOC dataset for main experiments. To install CLOC, 
please clone the dataset repo and follow the README instructions: https://github.com/IntelLabs/continuallearning/tree/main/CLOC

## Usage
This repo is based on the Avalanche framework. We highly recommend skimming over this quick tutorial to understand the basics of Avalanche:
https://avalanche.continualai.org/from-zero-to-hero-tutorial/01_introduction

The experiment folder is structred as follows:
```bash
experiments/<dataset_name>/<stream_name>
```

For example, fast stream experimens on CLOC are located at:
```bash
experiments/CLOC/fast_stream
```

To run experiments, simply activate the conda enviornment and run the experiment bash script in the desired dataset and stream folders.
