import os
import numpy as np
from numpy.lib.format import open_memmap

from tqdm import tqdm

sets = {'train', 'val'}
datasets = {'ntu/xview', 'ntu/xsub'}
parts = {'joint', 'bone'}


def gen_motion_data():
    for dataset in datasets:
        for set in sets:
            for part in parts:
                fn = '../data/{}/{}_data_{}.npy'.format(dataset, set, part)
                if not os.path.exists(fn):
                    print('Joint/bone data does not exist for {} {} set'.format(dataset, set))
                    continue

                print('Generating motion data for', dataset, set, part)
                data = np.load(fn)
                (N, C, T, V, M) = data.shape
                fp_sp = open_memmap(
                    '../data/{}/{}_data_{}_motion.npy'.format(dataset, set, part),
                    dtype='float32',
                    mode='w+',
                    shape=data.shape)

                # Loop through frames and insert motion difference
                for t in tqdm(range(T - 1)):
                    fp_sp[:, :, t, :, :] = data[:, :, t + 1, :, :] - data[:, :, t, :, :]

                # Pad last frame with 0
                fp_sp[:, :, T - 1, :, :] = 0


if __name__ == '__main__':
    gen_motion_data()