################################################################################
# Date: Nov-01-2022                                                            #
# Author(s): Yasir Ghunaim                                                     #
# E-mail: yasir.ghunaim@kaust.edu.sa                                           #
# Website: https://cemse.kaust.edu.sa/ivul                                     #
################################################################################
"""
This module contains the high-level CLOC (https://arxiv.org/abs/2108.09020) benchmark generator.
"""

from builtins import breakpoint
from pathlib import Path
from typing import Union, Any, Optional
from typing_extensions import Literal
from avalanche.benchmarks.utils.dataset_utils import ConstantSequence

from torchvision import transforms

from avalanche_extend.benchmarks.datasets.cloc import (
    CLOCDataset
)
from avalanche_extend.benchmarks.scenarios.generic_benchmark_creation_cloc import (
    create_generic_benchmark_from_paths_cloc,
)
from avalanche.benchmarks.classic.classic_benchmarks_utils import (
    check_vision_benchmark,
)

_default_train_transform = transforms.Compose(
    [
        transforms.Resize(256),
		transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ]
)

_default_eval_transform = transforms.Compose(
    [
        transforms.Resize(256),
		transforms.CenterCrop(224),
		transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ]
)


def CLOC(
    n_experiences=1,
    *,
    train_transform: Optional[Any] = _default_train_transform,
    eval_transform: Optional[Any] = _default_eval_transform,
    dataset_root="path/to/cloc_dataset/release/",
    debug = False,
    validation = False,
):
    """
    Creates an online stream benchmark for CLOC. This function assumes
    that the CLOC dataset is already installed in the "dataset_root".
    This benchmark reads a list of image paths to construct a dataset. 

    The benchmark instance returned by this method will have two fields,
    `train_stream` and `test_stream`, which can be iterated to obtain
    training and test :class:`Experience`. Since CLOC doesn't use task
    boundaries, training and test will contain only a single Experience.


    :param train_transform: The transformation to apply to the training data,
        e.g. a random crop, a normalization or a concatenation of different
        transformations (see torchvision.transform documentation for a
        comprehensive list of possible transformations). Defaults to CLOC train transform.
    :param eval_transform: The transformation to apply to the test data,
        e.g. a random crop, a normalization or a concatenation of different
        transformations (see torchvision.transform documentation for a
        comprehensive list of possible transformations). Defaults to CLOC test transform.
    :param dataset_root: The root path of the dataset.
        Defaults to None, which means that the default location for
        str(data_name) will be used.
    :param debug: If true, returns a small subset of CLOC (10,000 samples).
    :param validation: If ture, uses the validation set instead of training set.

    :returns: a properly initialized :class:`GenericCLScenario` instance.
    """


    cloc_dataset_train = CLOCDataset(
        split="train", debug=debug, dataset_root=dataset_root
    )
    cloc_dataset_valid = CLOCDataset(
        split="valid", debug=debug, dataset_root=dataset_root
    )
    cloc_dataset_test = CLOCDataset(
        split="test", debug=debug, dataset_root=dataset_root
    )
    
    if validation:
        train_samples, root = cloc_dataset_valid.get_paths_and_targets()
        print("Validation augmentation:")
        print(train_transform)
        # train_transform = eval_transform
    else:
        train_samples, root = cloc_dataset_train.get_paths_and_targets(n_experiences)
        print("\nTraining augmentation:")
        print(train_transform)
    
    test_samples, _ = cloc_dataset_test.get_paths_and_targets()
    print("\nTest augmentation:")
    print(eval_transform)
    print()

    benchmark_generator = create_generic_benchmark_from_paths_cloc

    benchmark_obj = benchmark_generator(
        train_samples,
        test_samples,
        task_labels=ConstantSequence(0, n_experiences),
        common_root=root,
        complete_test_set_only=False,
        train_transform=train_transform,
        eval_transform=eval_transform,
    )

    return benchmark_obj



__all__ = ["CLOC"]

if __name__ == "__main__":
    import sys
    from torch.utils.data import DataLoader
    
    benchmark_instance = CLOC()
    
    # check_vision_benchmark(benchmark_instance, show_without_transforms=False)
    # print(f"Check pass")
                
    sys.exit(0)
