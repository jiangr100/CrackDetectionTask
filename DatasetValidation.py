import pickle
from pathlib import Path
import os
import numpy as np
from PIL import Image
from hyperparameters import *
from models import *
from Dataset import *

if __name__ == '__main__':
    dataset = CrackClassification(
        data_dir=str(Path('D:\\pickle_files')),
        transform=None,
        mode='crack_type_group'
    )

    for idx, data in enumerate(dataset):
        print(idx)

    # print(cls)