import pickle
from pathlib import Path
import os
import numpy as np
from PIL import Image

if __name__ == '__main__':
    pickle_path = str(Path('D:\\pickle_files'))

    crack_to_cls = {
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

    cls = np.zeros(21)
    for f in os.listdir(pickle_path):
        with open(str(Path(pickle_path + '/' + f)), 'rb') as f:
            data = pickle.load(f, encoding='latin1')

        print(f, data)
        img = Image.open(str(Path(data['prefix'] + '/' + data['img_name'])))
        print("success")

        # for crack_type in data['crack_type']:
            # print(data)
            # cls[crack_to_cls[crack_type]] += 1

    """
    with open(str(Path(pickle_path + '/0240000')), 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    print(data)
    """

    # print(cls)