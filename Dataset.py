import torch
import os
from pathlib import Path
import imageio
import pickle
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, WeightedRandomSampler, Subset
from PIL import Image, ImageFile

class CrackPatches(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, crack_dir, pothole_dir, empty_dir, transform=None):
        self.crack_dir = crack_dir
        self.pothole_dir = pothole_dir
        self.empty_dir = empty_dir
        self.num_crack = len(os.listdir(crack_dir))
        self.num_pothole = len(os.listdir(pothole_dir)) if pothole_dir is not None else 0
        self.num_empty = len(os.listdir(empty_dir)) if empty_dir is not None else 0
        self.transform = transform

    def __len__(self):
        return self.num_crack + self.num_pothole + self.num_empty

    def __getitem__(self, idx):
        if idx < self.num_crack:
            dir_name = self.crack_dir
            file_name = "%06d.jpg" % idx
            label = 1
        elif idx < self.num_crack + self.num_pothole:
            dir_name = self.pothole_dir
            file_name = "%06d.jpg" % (idx - self.num_crack)
            label = 2
        else:
            dir_name = self.empty_dir
            file_name = "%06d.jpg" % (idx - self.num_crack - self.num_pothole)
            label = 0

        img = imageio.imread(str(Path(dir_name + '/' + file_name)))

        if self.transform is not None:
            img = self.transform(img)

        return {'img': img, 'y': label}

class CrackClassification(Dataset):
    def __init__(self, data_dir, mode, transform):
        self.data_dir = data_dir
        self.transform = transform
        self.mode = mode
        self.crack_to_cls = dict()
        self.crack_to_cls['cracks_with_sev'] = {
            'AL_L': 0,
            'AL_M': 1,
            'AL_H': 2,
            'BL_L': 3,
            'BL_M': 4,
            'BL_H': 5,
            'LO_LA_L': 6,
            'LO_LA_M': 7,
            'LO_LA_H': 8,
            'SW_L': 9,
            'SW_M': 10,
            'SW_H': 11,
            'PA_L': 12,
            'PA_M': 13,
            'PA_H': 14,
            'PO_L': 15,
            'PO_M': 16,
            'PO_H': 17,
            'ED_L': 18,
            'ED_M': 19,
            'ED_H': 20,
            'no_crack': 21
        }
        self.crack_to_cls['crack_type'] = {
            'AL': 0,
            'BL': 1,
            'LO_LA': 2,
            'SW': 3,
            'PA': 4,
            'PO': 5,
            'ED': 6,
            'no_crack': 7
        }
        self.crack_to_cls['crack_type_group'] = {
            'AL': 0,
            'BL': 0,
            'LO_LA': 1,
            'SW': 1,
            'PA': 2,
            'PO': 3,
            'ED': 1,
            'no_crack': 4
        }
        self.num_cls = len(set(self.crack_to_cls[mode].values()))

        self.weights = {'cracks_with_sev': dict(), 'crack_type': dict(), 'crack_type_group': dict()}
        self.crack_type_count = [3.8775e+04, 5.8387e+04, 1.3995e+04, 9.0370e+03, 1.6249e+04, 5.5280e+03,
                                 6.6281e+04, 4.0618e+04, 2.5270e+03, 9.0400e+03, 8.4460e+03, 8.3700e+02,
                                 7.1290e+03, 5.2310e+03, 1.0530e+03, 1.4290e+03, 3.4500e+02, 6.2000e+01,
                                 2.4220e+03, 4.5230e+03, 7.8300e+02, 4.0e+04]
        for i, (key, _) in enumerate(self.crack_to_cls['cracks_with_sev'].items()):
            self.weights['cracks_with_sev'].update({key: self.crack_type_count[i]})

        for i, (key, _) in enumerate(self.crack_to_cls['crack_type'].items()):
            if key != 'no_crack':
                self.weights['crack_type'].update({key: self.crack_type_count[i*3] +
                                                        self.crack_type_count[i*3+1] +
                                                        self.crack_type_count[i*3+2]})
            else:
                self.weights['crack_type'].update({key: self.crack_type_count[i*3]})

        for i, (key, _) in enumerate(self.weights['crack_type'].items()):
            if key == 'AL' or key == 'BL':
                self.weights['crack_type_group'].update({key: self.weights['crack_type']['AL'] +
                                                              self.weights['crack_type']['BL']})
            if key == 'LO_LA' or key == 'SW' or key == 'ED':
                self.weights['crack_type_group'].update({key: self.weights['crack_type']['LO_LA'] +
                                                              self.weights['crack_type']['SW'] +
                                                              self.weights['crack_type']['ED']})
            if key == 'PA':
                self.weights['crack_type_group'].update({key: self.weights['crack_type']['PA']})
            if key == 'PO':
                self.weights['crack_type_group'].update({key: self.weights['crack_type']['PO']})
            if key == 'no_crack':
                self.weights['crack_type_group'].update({key: self.weights['crack_type']['no_crack']})

        for key, _ in self.weights.items():
            for key2, _ in self.weights[key].items():
                self.weights[key][key2] = 1 / self.weights[key][key2]

        if os.path.exists(str(Path('sampler_weights_%s.pkl' % self.mode))):
            with open(str(Path('sampler_weights_%s.pkl' % self.mode)), 'rb') as f:
                self.sampler_weights = pickle.load(f, encoding='latin1')
        else:
            self.sampler_weights = []
            for pkl in os.listdir(self.data_dir):
                with open(str(Path(self.data_dir + '/' + pkl)), 'rb') as f:
                    data = pickle.load(f, encoding='latin1')
                load_field = 'cracks_with_sev' if self.mode == 'cracks_with_sev' else 'crack_type'
                current_weight = 0
                for crack_type in data[load_field]:
                    current_weight = max(current_weight, self.weights[self.mode][crack_type])
                self.sampler_weights.append(current_weight)

            with open(str(Path('sampler_weights_%s.pkl' % self.mode)), 'wb') as f:
                pickle.dump(self.sampler_weights, f)

        ImageFile.LOAD_TRUNCATED_IMAGES = True

    def __len__(self):
        return len(os.listdir(self.data_dir))

    def __getitem__(self, idx):
        with open(str(Path(self.data_dir + "/%07d"%idx)), 'rb') as f:
            data = pickle.load(f, encoding='latin1')

        img = Image.open(str(Path(data['prefix'] + '/' + data['img_name'])))
        cls = torch.zeros(self.num_cls)
        load_field = 'cracks_with_sev' if self.mode == 'cracks_with_sev' else 'crack_type'
        for crack_type in data[load_field]:
            cls[self.crack_to_cls[self.mode][crack_type]] = 1

        return {'img': img if self.transform == None else self.transform(img), 'y': cls}

def create_dataloader(args, dataset, test_split=0.2, validation_split=0.01, with_weight=False):
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
    if with_weight:
        weights = dataset.sampler_weights
        print()
        train_dataset = Subset(dataset, train_indices)
        train_weights = [weights[i] for i in train_indices]
        train_sampler = WeightedRandomSampler(train_weights, len(train_weights))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    else:
        train_sampler = SubsetRandomSampler(train_indices)
        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)

    valid_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    validation_loader = DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler)
    test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)

    return train_loader, validation_loader, test_loader
