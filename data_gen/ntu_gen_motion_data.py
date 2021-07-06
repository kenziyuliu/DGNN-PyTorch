import os
import numpy as np
from numpy.lib.format import open_memmap

from tqdm import tqdm

sets = {'train', 'val'}
datasets = {'ntu/xview', 'ntu/xsub'}
parts = {'joint', 'bone'}

def gen_motion_data():
    for dataset in datasets: # 'ntu/xview', 'ntu/xsub'
        for set in sets: # 'train', 'val'
            for part in parts: # 'joint', 'bone'
                fn = './data/{}/{}_data_{}.npy'.format(dataset, set, part)
                
                # Joint, Bone 파일 존재하지 않으면 for문 continue 실행
                if not os.path.exists(fn):
                    print('Joint/bone data does not exist for {} {} set'.format(dataset, set))
                    continue

                # Joint, Bone 파일이 존재하면 
                print('Generating motion data for', dataset, set, part)
                
                # Joint, Bone 데이터 Load
                data = np.load(fn)
                (N, C, T, V, M) = data.shape
                fp_sp = open_memmap(
                    './data/{}/{}_data_{}_motion.npy'.format(dataset, set, part),
                    dtype='float32',
                    mode='w+',
                    shape=data.shape)

                # Motion데이터는 dt동안 Joint가 움직인 거리. m = v(t+1) - v(t) 
                # Loop through frames and insert motion difference
                for t in tqdm(range(T - 1)):
                    fp_sp[:, :, t, :, :] = data[:, :, t + 1, :, :] - data[:, :, t, :, :]

                # 마지막 한 프레임은 0으로 채움
                fp_sp[:, :, T - 1, :, :] = 0


if __name__ == '__main__':
    gen_motion_data()