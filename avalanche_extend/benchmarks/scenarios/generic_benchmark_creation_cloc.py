################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: Nov-01-2022                                                            #
# Author(s): Yasir Ghunaim                                                     #
# E-mail: yasir.ghunaim@kaust.edu.sa                                           #
# Website: https://cemse.kaust.edu.sa/ivul                                     #
################################################################################

""" This file defines a special benchmark generator for the CLOC dataset"""

from pathlib import Path
from typing import (
    Sequence,
    Union,
    Any,
    Tuple,
    Dict
)

from avalanche.benchmarks.utils import (
    AvalancheDataset,
    PathsDataset,
)
from avalanche.benchmarks.scenarios.classification_scenario import GenericCLScenario
from avalanche.benchmarks.utils.avalanche_dataset import AvalancheDatasetType
from avalanche.benchmarks.scenarios.generic_benchmark_creation import create_multi_dataset_generic_benchmark, FileAndLabel

FileLabelAndIndex = Union[
    Tuple[Union[str, Path], int, int], Tuple[Union[str, Path], int, int, Sequence]
]

class PathsIndexedDataset(PathsDataset):
    """
    This class extends the PathsDataset class which creates a File Dataset from a list of files and labels.
    
    PathsIndexedDataset extends PathsDataset by adding an index element with each file path to maintain
    a chronological reference for images in the stream. 
    
    """

    def __getitem__(self, index):
        """
        Returns next element in the dataset given the current index.

        :param index: index of the data to get.
        :return: loaded item.
        """

        img_description = self.imgs[index]
        impath = img_description[0]
        target = img_description[1]
        original_index = img_description[2]
        bbox = None
        if len(img_description) > 3:
            bbox = img_description[3]

        if self.root is not None:
            impath = self.root / impath
        img = self.loader(impath)

        # If a bounding box is provided, crop the image before passing it to
        # any user-defined transformation.
        if bbox is not None:
            if isinstance(bbox, Tensor):
                bbox = bbox.tolist()
            img = crop(img, *bbox)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, original_index

    def __len__(self):
        """
        Returns the total number of elements in the dataset.

        :return: Total number of dataset items.
        """

        return len(self.imgs)

def create_generic_benchmark_from_paths_cloc(
    train_lists_of_files: Sequence[Sequence[FileLabelAndIndex]],
    test_lists_of_files: Union[
        Sequence[FileLabelAndIndex], Sequence[Sequence[FileLabelAndIndex]]
    ],
    *,
    other_streams_lists_of_files: Dict[
        str, Sequence[Sequence[FileAndLabel]]
    ] = None,
    task_labels: Sequence[int],
    common_root: None,
    complete_test_set_only: bool = False,
    train_transform=None,
    train_target_transform=None,
    eval_transform=None,
    eval_target_transform=None,
    other_streams_transforms: Dict[str, Tuple[Any, Any]] = None,
    dataset_type: AvalancheDatasetType = AvalancheDatasetType.UNDEFINED
) -> GenericCLScenario:
    """
    This is a special implementation for CLOC benchmark based on the original 
    implementation of 'create_generic_benchmark_from_paths'. The difference 
    is that this method accepts common_root as a parameter instead of extracting
    it from the list of paths. Since CLOC dataset is huge, this method is much
    faster compared to the original implementation.

    Creates a benchmark instance given a sequence of lists of files. A separate
    dataset will be created for each list. Each of those datasets
    will be considered a separate experience.

    This is very similar to :func:`create_generic_benchmark_from_filelists`,
    with the main difference being that
    :func:`create_generic_benchmark_from_filelists` accepts, for each
    experience, a file list formatted in Caffe-style. On the contrary, this
    accepts a list of tuples where each tuple contains two elements: the full
    path to the pattern and its label. Optionally, the tuple may contain a third
    element describing the bounding box of the element to crop. This last
    bounding box may be useful when trying to extract the part of the image
    depicting the desired element.

    Apart from that, the same limitations of
    :func:`create_generic_benchmark_from_filelists` regarding task labels apply.

    The label of each pattern doesn't have to be an int. Also, a dataset type
    can be defined.

    :param train_lists_of_files: A list of lists. Each list describes the paths,
        labels and index of patterns to include in that training experience, as
        tuples. Each tuple must contain three elements: the full path to the
        pattern, its class label and index. Optionally, the tuple may contain a
        fourth element describing the bounding box to use for cropping (top,
        left, height, width).
    :param test_lists_of_files: A list of lists. Each list describes the paths,
        labels and index of patterns to include in that test experience, as tuples.
        Each tuple must contain three elements: the full path to the pattern,
        its class label and index. Optionally, the tuple may contain a fourth element
        describing the bounding box to use for cropping (top, left, height,
        width).
    :param other_streams_lists_of_files: A dictionary describing the content of
        custom streams. Keys must be valid stream names (letters and numbers,
        not starting with a number) while the value follow the same structure
        of `train_lists_of_files` and `test_lists_of_files` parameters. If this
        dictionary contains the definition for "train" or "test" streams then
        those definition will  override the `train_lists_of_files` and
        `test_lists_of_files` parameters.
    :param task_labels: A list of task labels. Must contain at least a value
        for each experience. Each value describes the task label that will be
        applied to all patterns of a certain experience. For more info on that,
        see the function description.
    :param common_root: The absolute path of the common root for the list of paths.
    :param complete_test_set_only: If True, only the complete test set will
        be returned by the benchmark. This means that the ``test_list_of_files``
        parameter must define a single experience (the complete test set).
        Defaults to False.
    :param train_transform: The transformation to apply to the training data,
        e.g. a random crop, a normalization or a concatenation of different
        transformations (see torchvision.transform documentation for a
        comprehensive list of possible transformations). Defaults to None.
    :param train_target_transform: The transformation to apply to training
        patterns targets. Defaults to None.
    :param eval_transform: The transformation to apply to the test data,
        e.g. a random crop, a normalization or a concatenation of different
        transformations (see torchvision.transform documentation for a
        comprehensive list of possible transformations). Defaults to None.
    :param eval_target_transform: The transformation to apply to test
        patterns targets. Defaults to None.
    :param other_streams_transforms: Transformations to apply to custom
        streams. If no transformations are defined for a custom stream,
        then "train" transformations will be used. This parameter must be a
        dictionary mapping stream names to transformations. The transformations
        must be a two elements tuple where the first element defines the
        X transformation while the second element is the Y transformation.
        Those elements can be None. If this dictionary contains the
        transformations for "train" or "test" streams then those transformations
        will override the `train_transform`, `train_target_transform`,
        `eval_transform` and `eval_target_transform` parameters.
    :param dataset_type: The type of the dataset. Defaults to UNDEFINED.

    :returns: A :class:`GenericCLScenario` instance.
    """
    
    assert common_root is not None, "Please specify a root path."

    input_streams = dict(train=train_lists_of_files, test=test_lists_of_files)

    if other_streams_lists_of_files is not None:
        input_streams = {**input_streams, **other_streams_lists_of_files}

    stream_definitions = dict()

    
    for stream_name, lists_of_files in input_streams.items():
        stream_datasets = []
        for exp_id, list_of_files in enumerate(lists_of_files):
            paths_dataset = PathsIndexedDataset(common_root, list_of_files)
            stream_datasets.append(
                AvalancheDataset(paths_dataset, task_labels=task_labels[exp_id])
            )

        stream_definitions[stream_name] = stream_datasets

    return create_multi_dataset_generic_benchmark(
        [],
        [],
        other_streams_datasets=stream_definitions,
        train_transform=train_transform,
        train_target_transform=train_target_transform,
        eval_transform=eval_transform,
        eval_target_transform=eval_target_transform,
        complete_test_set_only=complete_test_set_only,
        other_streams_transforms=other_streams_transforms,
        dataset_type=dataset_type,
    )



__all__ = [
    "create_generic_benchmark_from_paths_cloc"
]
