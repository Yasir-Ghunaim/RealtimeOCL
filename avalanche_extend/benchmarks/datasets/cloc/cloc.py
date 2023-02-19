################################################################################
# Date: Nov-01-2022                                                            #
# Author: Yasir Ghunaim                                                        #
# E-mail: yasir.ghunaim@kaust.edu.sa                                           #
# Website: https://cemse.kaust.edu.sa/ivul                                     #
################################################################################

"""CLOC Pytorch Dataset
Dataset can be installed from: https://github.com/IntelLabs/continuallearning/tree/main/CLOC
Paper: "Online Continual Learning with Natural Distribution Shifts: An Empirical Study with Visual Data."
https://arxiv.org/abs/2108.09020
"""

# from typing import Any, List
import os
import logging
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import torch

from torchvision.datasets.folder import default_loader

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm',
                  '.tif', '.tiff', '.webp')


# https://stackoverflow.com/a/2135920
def split_range(a, n):
    k, m = divmod(len(a), n)
    return (list(a[i*k+min(i, m):(i+1)*k+min(i+1, m)]) for i in range(n))

class CLOCDataset(Dataset):
    """CLOC Pytorch Dataset
    - Training Images in total: 38,003,083
    - Corss Validation Images in total: 2,064,152
    - Test Images in total: 383,229
    - Shape of images: torch.Size([1, 3, 640, 480])
    """

    splits = ["train", "valid", "test"]

    def __init__(
        self,
        dataset_root = "path/to/cloc_dataset/release/",
        dataset_path = "dataset/images/",
        split="train",
        transform=ToTensor(),
        target_transform=None,
        loader=default_loader,
        extensions=IMG_EXTENSIONS,
        debug=False,
    ):
        """
        :param debug: If true, returns a small subset of CLOC (10,000 samples).
        """
        super().__init__()
        assert split in self.splits
        self.split = split
        self.debug = debug


        root = dataset_root + dataset_path

        if self.split == "train" or self.split == "valid":
            fname = dataset_root
        else:
            fname = dataset_root + \
                "yfcc100m_metadata_with_labels_usedDataRatio0." + \
                    "05_t110000_t250_valid_files_2004To2014_compact_val.csv"

        if isinstance(fname, torch._six.string_classes):
            fname = os.path.expanduser(fname)
        self.fname = fname

        self.transform = transform
        self.target_transform = target_transform
        self.root = root
        self.loader = loader
        self.extensions = extensions

        self._make_data()
        self._set_data_idx()

    def _make_data(self):
        if self.split == "train":
            # train set
            self.labels = torch.load(self.fname+'train_labels.torchSave')
            self.time_taken = torch.load(self.fname+'train_time.torchSave')
            self.user = torch.load(self.fname+'train_userID.torchSave')
            self.store_loc = torch.load(self.fname+'train_store_loc.torchSave')
            self.store_loc = list(map(lambda s: s.strip(), self.store_loc))
        elif self.split == "valid":
            # cross validation set
            self.labels = torch.load(self.fname+'cross_val_labels.torchSave')
            self.time_taken = torch.load(self.fname+'cross_val_time.torchSave')
            self.user = torch.load(self.fname+'cross_val_user.torchSave')
            self.store_loc = torch.load(self.fname+'cross_val_store_loc.torchSave')
            self.store_loc = list(map(lambda s: s.strip(), self.store_loc))
        else:
            # test set
            fval = open(self.fname, 'r')
            lines_val = fval.readlines()
            labels = [None] * len(lines_val)
            time = [None] * len(lines_val)
            user = [None] * len(lines_val)
            store_loc = [None] * len(lines_val)
            for i in range(len(lines_val)):
                line_splitted = lines_val[i].split(",")
                labels[i] = int(line_splitted[0])
                time[i] = int(line_splitted[2])
                user[i] = line_splitted[3]
                store_loc[i] = line_splitted[-1][:-1]

            self.labels = labels
            self.time_taken = time
            self.user = user
            self.store_loc = store_loc
        
        if self.debug:
            self.labels = self.labels[:10000]
            self.store_loc = self.store_loc[:10000]

    def _set_data_idx(self):
        self.data_size = len(self.labels)
        self.data_idx = list(range(0, self.data_size))

    def get_paths_and_targets(self, n_experiences=1):
        """Return a tuple of paths, targets and indices in addition to the root path"""
        paths_and_targets = []
        if n_experiences == 1:
            paths_and_targets.append(list(zip(self.store_loc, self.labels, self.data_idx)))
        else:
            subIndexList = list(split_range(range(len(self.store_loc)), n_experiences))
            for exp_id in range(n_experiences):
                paths_and_targets.append(list(zip(
                    [self.store_loc[i] for i in subIndexList[exp_id]],
                    [self.labels[i] for i in subIndexList[exp_id]],
                    [i for i in subIndexList[exp_id]])))

        return paths_and_targets, self.root

    def __getitem__(self, index):
        index = self.data_idx[index] 
        index_pop = index

        if index_pop < 0:
            index_pop = 0
            is_valid = torch.tensor(0)
        else:
            is_valid = torch.tensor(1)

        target = self.labels[index_pop]
        path = self.root + self.store_loc[index_pop]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.data_idx)


    
if __name__ == "__main__":
    # this litte example script can be used to visualize the first image
    # leaded from the dataset.
    from torch.utils.data.dataloader import DataLoader
    import matplotlib.pyplot as plt
    from torchvision import transforms
    import torch

    train_data = CLOCDataset()
    valid_data = CLOCDataset(split="valid")
    test_data = CLOCDataset(split="test")
    print("train size: ", len(train_data))
    print("valid size: ", len(valid_data))
    print("test size: ", len(test_data))

    dataloader = DataLoader(train_data, batch_size=1)

    for batch_data in dataloader:
        x, y = batch_data
        plt.imshow(transforms.ToPILImage()(torch.squeeze(x)))
        plt.show()
        print(x.size())
        print(len(y))
        break

__all__ = ["CLOCDataset"]
