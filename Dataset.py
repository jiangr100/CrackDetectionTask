import torch
import os
from pathlib import Path
import imageio
import pickle
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler

class CrackPatches(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, positive_root_dir, negative_root_dir, transform=None):
        self.positive_root_dir = positive_root_dir
        self.negative_root_dir = negative_root_dir
        self.num_positive = len(os.listdir(positive_root_dir))
        self.num_negative = len(os.listdir(negative_root_dir))
        self.transform = transform

    def __len__(self):
        return self.num_positive + self.num_negative

    def __getitem__(self, idx):
        if idx < self.num_positive:
            dir_name = self.positive_root_dir
            file_name = "%05d.jpg" % idx
            label = 1
        else:
            dir_name = self.negative_root_dir
            file_name = "%05d.jpg" % (idx - self.num_positive)
            label = 0

        img = imageio.imread(str(Path(dir_name + '/' + file_name)))

        if self.transform is not None:
            img = self.transform(img)

        return {'img': img, 'y': label}


def create_dataloader(args, dataset, test_split=0.2, validation_split=0.01):
    batch_size = args.batch_size
    shuffle_dataset = True
    random_seed = args.random_seed

    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split_1 = int(np.floor(test_split * dataset_size))
    split_2 = int(np.floor((test_split+validation_split) * dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices, test_indices = indices[split_2:], indices[split_1:split_2], indices[:split_1]

    print(len(train_indices))

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    validation_loader = DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler)
    test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)

    return train_loader, validation_loader, test_loader
