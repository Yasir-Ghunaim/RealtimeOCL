# Real-Time Evaluation in Online Continual Learning: A New Hope (CVPR2023, Highlight)
A realistic and fair evaluation for Online Continual Learning (OCL), in which the computational cost of OCL methods is incorporated into the training process. That is, more expensive OCL methods than our baseline ends up training on proportionally fewer data.

#### This work was accepted at CVPR 2023 as a highlight (top 2.5%) paper. [Link to preprint](https://arxiv.org/abs/2302.01047)

## Requirements
Install the conda enviornment by running: 
```bash
    conda create -n realtime_ocl python=3.9
    conda activate realtime_ocl
    pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
    pip install avalanche-lib==0.2.0 deepspeed==0.6.7 tensorboardx
```

## CLOC Installation
For our main experiments, we rely on the CLOC dataset. To install CLOC, please clone the dataset repository and follow the instructions provided in the README file, which can be found at the following link: https://github.com/IntelLabs/continuallearning/tree/main/CLOC

## CLOC Preprocessing
When downloading CLOC images, it is necessary to access a large list of image URLs hosted on Flickr server. However, due to potential issues such as changes in the availability of the images, it is possible that some images will be missing after the download process is complete. To ensure optimal training time and minimize errors during dataloading, we strongly recommend removing these invalid pointers from the CLOC metadata. This can be accomplished by following the steps outlined in the "preprocess_CLOC.ipynb" notebook.

## Usage
This repo is based on the Avalanche framework. We highly recommend going over this quick tutorial to understand the basics of Avalanche:
https://avalanche.continualai.org/from-zero-to-hero-tutorial/01_introduction

The experiment folder is structred as follows:
```bash
experiments/<dataset_name>/<stream_name>
```

For example, fast stream experimens on CLOC are located at:
```bash
experiments/CLOC/fast_stream
```

To run experiments, simply activate the conda enviornment and run the experiment bash script in the desired dataset/stream folder.

## Cite
```
@inproceedings{ghunaim2023real,
  title={Real-Time Evaluation in Online Continual Learning: A New Hope},
  author={Ghunaim, Yasir and Bibi, Adel and Alhamoud, Kumail and Alfarra, Motasem and Hammoud, Hasan Abed Al Kader and Prabhu, Ameya and Torr, Philip HS and Ghanem, Bernard},
  booktitle = {Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2023}
}
```
