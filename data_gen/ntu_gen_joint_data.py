import os
import re
import pickle
import argparse
import numpy as np

import multiprocessing
from tqdm import tqdm
import sys
sys.path.extend(['../'])
from data_gen.preprocess import pre_normalization

# For Cross-Subject benchmark "xsub"
training_subjects = [1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38]
# For Cross-View benchmark "xview"
training_cameras = [2, 3]

max_body_true = 2
max_body_kinect = 4
num_joint = 25
max_frame = 300


def read_skeleton_filter(file):
    with open(file, 'r') as f:
        skeleton_sequence = {}
        skeleton_sequence['numFrame'] = int(f.readline())
        skeleton_sequence['frameInfo'] = []
        # num_body = 0
        for t in range(skeleton_sequence['numFrame']):
            frame_info = {}
            frame_info['numBody'] = int(f.readline())
            frame_info['bodyInfo'] = []

            for m in range(frame_info['numBody']):
                body_info = {}
                body_info_key = [
                    'bodyID', 'clipedEdges', 'handLeftConfidence',
                    'handLeftState', 'handRightConfidence', 'handRightState',
                    'isResticted', 'leanX', 'leanY', 'trackingState'
                ]
                body_info = {
                    k: float(v)
                    for k, v in zip(body_info_key, f.readline().split())
                }
                body_info['numJoint'] = int(f.readline())
                body_info['jointInfo'] = []
                for v in range(body_info['numJoint']):
                    joint_info_key = [
                        'x', 'y', 'z', 'depthX', 'depthY', 'colorX', 'colorY',
                        'orientationW', 'orientationX', 'orientationY',
                        'orientationZ', 'trackingState'
                    ]
                    joint_info = {
                        k: float(v)
                        for k, v in zip(joint_info_key, f.readline().split())
                    }
                    body_info['jointInfo'].append(joint_info)
                frame_info['bodyInfo'].append(body_info)
            skeleton_sequence['frameInfo'].append(frame_info)

    return skeleton_sequence


def get_nonzero_std(s):
    # `s` has shape (T, V, C)
    # Select valid frames where sum of all nodes is nonzero
    s = s[s.sum((1,2)) != 0]
    if len(s) != 0:
        # Compute sum of standard deviation for all 3 channels as `energy`
        s = s[..., 0].std() + s[..., 1].std() + s[..., 2].std()
    else:
        s = 0
    return s


def read_xyz(file, max_body=4, num_joint=25):  # 取了前两个body
    seq_info = read_skeleton_filter(file)
    # Create data tensor of shape: (# persons (M), # frames (T), # nodes (V), # channels (C))
    data = np.zeros((max_body, seq_info['numFrame'], num_joint, 3))
    for n, f in enumerate(seq_info['frameInfo']):
        for m, b in enumerate(f['bodyInfo']):
            for j, v in enumerate(b['jointInfo']):
                if m < max_body and j < num_joint:
                    data[m, n, j, :] = [v['x'], v['y'], v['z']]

    # select 2 max energy body
    energy = np.array([get_nonzero_std(x) for x in data])
    index = energy.argsort()[::-1][0:max_body_true]
    data = data[index]
    # Data new shape: (C, T, V, M)
    data = data.transpose(3, 1, 2, 0)
    return data


def gendata(data_path, out_path, ignored_sample_path=None, benchmark='xview', part='eval'):
    if ignored_sample_path != None:
        with open(ignored_sample_path, 'r') as f:
            ignored_samples = [line.strip() + '.skeleton' for line in f.readlines()]
    else:
        ignored_samples = []

    sample_name = []
    sample_label = []
    for filename in os.listdir(data_path):
        if filename in ignored_samples:
            continue
        action_class = int(filename[filename.find('A') + 1:filename.find('A') + 4])
        subject_id = int(filename[filename.find('P') + 1:filename.find('P') + 4])
        camera_id = int(filename[filename.find('C') + 1:filename.find('C') + 4])

        if benchmark == 'xview':
            istraining = (camera_id in training_cameras)
        elif benchmark == 'xsub':
            istraining = (subject_id in training_subjects)
        else:
            raise ValueError('Invalid benchmark provided: {}'.format(benchmark))

        if part == 'train':
            issample = istraining
        elif part == 'val':
            issample = not (istraining)
        else:
            raise ValueError('Invalid data part provided: {}'.format(part))

        if issample:
            sample_name.append(filename)
            sample_label.append(action_class - 1)

    with open('{}/{}_label.pkl'.format(out_path, part), 'wb') as f:
        pickle.dump((sample_name, list(sample_label)), f)

    # Create data tensor with shape (# examples (N), C, T, V, M)
    fp = np.zeros((len(sample_label), 3, max_frame, num_joint, max_body_true), dtype=np.float32)

    # Fill in the data tensor `fp` with training examples in parallel
    with multiprocessing.Pool() as p:
        results = list(tqdm(p.imap(read_xyz, [os.path.join(data_path, s) for s in sample_name]), total=len(sample_name)))
        for i, data in enumerate(results):
            fp[i, :, :data.shape[1], :, :] = data

    fp = pre_normalization(fp)
    np.save('{}/{}_data_joint.npy'.format(out_path, part), fp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NTU-RGB-D Data Converter.')
    parser.add_argument('--data_path', default='../data/nturgbd_raw/nturgb+d_skeletons/')
    parser.add_argument('--ignored_sample_path',
                        default='../data/nturgbd_raw/samples_with_missing_skeletons.txt')
    parser.add_argument('--out_folder', default='../data/ntu/')

    benchmarks = ['xsub', 'xview']
    parts = ['train', 'val']
    arg = parser.parse_args()

    for b in benchmarks:
        for p in parts:
            out_path = os.path.join(arg.out_folder, b)
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            print(b, p)
            gendata(
                arg.data_path,
                out_path,
                arg.ignored_sample_path,
                benchmark=b,
                part=p)
