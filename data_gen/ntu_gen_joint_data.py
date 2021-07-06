import os
import re
import pickle
import argparse
import numpy as np
import yaml

from tqdm import tqdm
import sys
sys.path.extend(['../'])
from preprocess import pre_normalization

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

# 관절의 x,y,z 읽어오기
def read_xyz(file, max_body=4, num_joint=25): 
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

    # Data new shape: (C, T, V, M) (# channels (C), # frames (T), # nodes (V), # persons (M))
    data = data.transpose(3, 1, 2, 0)
    return data


def gendata(data_path, out_path, node_mask=None, attention_action=None, ignored_sample_path=None, 
            benchmark='xview', part='eval'):
    
    ########################################### 데이터 선별 ###########################################
    # attention action, ignored_sample_path, benchmark, part에 따라 Train 데이터와 Test 데이터를 선별 
    # 선별된 데이터의 파일 이름과 Label을 List로 저장
    ###################################################################################################
    # ignored_sample_path
    if ignored_sample_path != None:
        with open(ignored_sample_path, 'r') as f:
            ignored_samples = [line.strip() + '.skeleton' for line in f.readlines()]
    else:
        ignored_samples = []

    sample_name = [] # benchmark와 part에 따라 선택되는 데이터 파일의 이름을 저장하는 List
    sample_label = [] # benchmark와 part에 따라 선택되는 데이터 파일의 라벨을 저장하는 List
    
    # Label data 생성
    for filename in os.listdir(data_path):
        # ignored_sample_path가 입력으로 들어오면 그 데이터는 Pass
        if filename in ignored_samples:
            continue
        
        # attention_action에 입력이 있으면 그 action class만 데이터 생성
        if attention_action != None :
            if filename[filename.find('A'):filename.find('A') + 4] not in attention_action :
                continue
        
        action_class = int(filename[filename.find('A') + 1:filename.find('A') + 4])
        subject_id = int(filename[filename.find('P') + 1:filename.find('P') + 4])
        camera_id = int(filename[filename.find('C') + 1:filename.find('C') + 4])

        # View와 Sub의 Train 데이터셋 분리하는 부분
        if benchmark == 'xview':
            istraining = (camera_id in training_cameras) # istraining은 bool type
        elif benchmark == 'xsub':
            istraining = (subject_id in training_subjects) # istraining은 bool type
        else:
            raise ValueError('Invalid benchmark provided: {}'.format(benchmark))

        # Train과 Val에 따라 데이터 선택
        if part == 'train':
            issample = istraining
        elif part == 'val':
            issample = not (istraining)
        else:
            raise ValueError('Invalid data part provided: {}'.format(part))

        # 선택된 Sample들의 Filename과 라벨 데이터를 list로 저장
        if issample: 
            sample_name.append(filename)
            
            # attention action이 있으면, 라벨들이 앞으로 당겨지게끔 설정
            if attention_action != None :
                sample_label.append(attention_action.index(filename[filename.find('A'):filename.find('A') + 4]))
            
            # attention action이 없으면, filename 안의 Action number가 라벨
            else:
                sample_label.append(action_class - 1)
                
    # 선택된 Sample들의 이름과 라벨 저장
    with open('{}/{}_label.pkl'.format(out_path, part), 'wb') as f:
        pickle.dump((sample_name, list(sample_label)), f)

    ########################################## 데이터 전처리 ##########################################
    # 위에서 선별된 File name에 맞는 데이터를 Load하여 N, C, T, V, M의 joint 데이터를 Load
    # Joint data Normalize
    # node_mask로 선택한 node만 npy 파일로 데이터 저장
    ###################################################################################################
    # Joint data 생성    
    # Joint data를 저장할 임시 변수 생성. tensor with shape (# examples (N), C=3, T=300, V=25, M=2)
    fp = np.zeros((len(sample_label), 3, max_frame, num_joint, max_body_true), dtype=np.float32)

    # Sample File에서 Joint Data를 읽어와서 임시 변수(fp)에 저장
    for i, s in enumerate(tqdm(sample_name)):
        # Data new shape: (C, T, V, M)
        data = read_xyz(os.path.join(data_path, s), max_body=max_body_kinect, num_joint=num_joint) # (C, T, V, M)
        fp[i, :, :data.shape[1], :, :] = data # (N, C, T, V, M)
    
    # fp normalize
    fp = pre_normalization(fp) # (N, C=3, T=300, V=25, M=2)
    
    
    # 특정 노드만 가져오기
    if node_mask != None : 
        node_mask = node_mask - np.ones([len(node_mask)], dtype=int)
        fp = fp[:, :, :, node_mask, :]
    
    # Joint data 저장
    np.save('{}/{}_data_joint.npy'.format(out_path, part), fp)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NTU-RGB-D Data Converter.')
    parser.add_argument('--data_path', default='/mnt/disk2/data/private_data/NTU_RGB+D/nturgb+d_skeletons/nturgbd_skeletons_s001_to_s017/')
    parser.add_argument('--ignored_sample_path',
                        default='./data/nturgbd_raw/samples_with_missing_skeletons.txt')
    parser.add_argument('--out_folder', default='./data/ntu/')
    parser.add_argument(
            '--node_mask',
            default=None,
            help='사용할 노드') # 사용할 노드 
    parser.add_argument(
            '--selected_action',
            default=None,
            help='action to train') # 사용할 Action 번호
    parser.add_argument(
            '--config',
            default='./config/node_config.yaml',
            help='path to the configuration file') # config 파일이 위치하는 디렉토리

    benchmarks = ['xsub', 'xview']
    parts = ['train', 'val']
    arg = parser.parse_args()

    if arg.config is not None: 
        with open(arg.config, 'r') as f:
            default_arg = yaml.load(f) # config 파일에 들어있는 keys. config key로 명명
 
        # 예외 처리 
        key = vars(arg).keys() # parser에 들어있는 key값. default key로 명명
        for k in default_arg.keys():
            if k not in key: # config key가 default key에 들어있지 않으면
                print('WRONG ARG: {}'.format(k)) # default key에 해당 키가 없음을 알림.
                assert (k in key)

        parser.set_defaults(**default_arg) # 임의의 개수의 Keyword arguments를 받아서 default key -> config key로 변경

    arg = parser.parse_args() # config key가 반영된 argument

    print("arg : ", arg)

    for b in benchmarks:
        for p in parts:
            out_path = os.path.join(arg.out_folder, b)
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            print(b, p)
            gendata(
                arg.data_path, # 전처리할 Skeleton data가 위치한 경로. NTU RGB 데이터를 의미
                out_path, # 전처리된 데이터가 위치할 공통 경로.
                node_mask=arg.node_mask, # node mask. 25개의 Joint를 다 사용하지 않을 경우 사용할 Joint를 List 형태로 넣으면 된다. 
                attention_action = arg.selected_action, # 특정 Action에 대해서만 데이터 생성. 원하는 Action을 List 형태로 넣으면 된다.
                ignored_sample_path = arg.ignored_sample_path, # 무시할 Sample 데이터가 적혀진 txt파일의 경로. 
                benchmark=b, # xsub / xview
                part=p) # train할지, validation할지 
