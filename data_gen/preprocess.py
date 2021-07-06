import sys
sys.path.extend(['../'])

import numpy as np
from rotation import angle_between, rotation_matrix
from tqdm import tqdm

def pre_normalization(data, zaxis=[0, 1], xaxis=[8, 4]):
    # examples (N), channels (C), frames (T), nodes (V), persons (M))
    N, C, T, V, M = data.shape
    s = np.transpose(data, [0, 4, 2, 3, 1])  # to (N, M, T, V, C)
    
    ###################################################################################################################
    # 1. 스켈레톤 데이터에서 중간 Frame부터 동작이 있는 경우 데이터 Shift
    # 2. 스켈레톤 데이터에서 뒷부분에 동작이 없는 경우, 유효한 데이터를 여러번 반복
    print('pad the null frames with the previous frames')
    for i_s, skeleton in enumerate(tqdm(s)):  # Dimension N   # skeleton : (M, T, V, C) (2, 300, 25, 3)
        
        # skeleton 데이터가 전부 0으로 되어 있는 데이터 찾기. - 잘못된 데이터 검출 
        # continue가 빠진 것 같음
        if skeleton.sum() == 0:  
            print(i_s, ' has no skeleton')
        
        # 전체 스켈레톤 데이터에 대해
        for i_p, person in enumerate(skeleton): # Dimension M (# person)  # person : (T, V, C) (300, 25, 3)
            # Person 데이터가 전부 0인 경우(Person 1과 Person 2가 전부 0)이면 넘기기
            if person.sum() == 0: # person 
                continue
            
            # 1. Person의 첫번째 프레임이 0인 경우. 즉, 프레임 초반에는 데이터가 없는 경우.      
            if person[0].sum() == 0:
                # `index` of frames that have non-zero nodes
                # frames의 합 != 0인 인덱스 찾기. 즉, T가 30일 때, Frame이 5부터 10까지인 데이터는 5~10의 인덱스 반환.
                index = (person.sum(-1).sum(-1) != 0)
                tmp = person[index].copy() # 프레임 임시 저장
                # Shift non-zero nodes to beginning of frames
                person *= 0
                person[:len(tmp)] = tmp # shift
            
            # 전체 스켈레톤 데이터의 Node와 Channel에 대해
            for i_f, frame in enumerate(person): # Each frame : (V, C)
                # 한 프레임에 대해서 데이터가 0이면(스켈레톤 데이터가 없으면)
                if frame.sum() == 0:
                    # 그 프레임의 데이터부터 끝까지 데이터가 0이면(스켈레톤 데이터가 없으면)
                    if person[i_f:].sum() == 0:
                        # 2. Repeat all the frames up to now (`i_f`) till the max seq len
                        rest = len(person) - i_f # 나머지 프레임 갯수
                        reps = int(np.ceil(rest / i_f)) # 같은 동작을 몇번 반복할 것인지.
                        pad = np.concatenate([person[:i_f] for _ in range(reps)], 0)[:rest] # 프레임 수가 넘어가면 뒷부분 자르기
                        s[i_s, i_p, i_f:] = pad
                        break
    ###################################################################################################################
    
    # Joint 데이터를 x, y, z = 0, 0, 0 좌표 근처로 이동
    print('subtract the center joint #1 (spine joint in ntu and neck joint in kinetics)')
    for i_s, skeleton in enumerate(tqdm(s)):
        # Person 데이터가 전부 0인 경우(Person 1과 Person 2가 전부 0)이면 넘기기
        if skeleton.sum() == 0:
            continue
            
        # Use the first skeleton's body center (`1:2` along the nodes dimension)  
        # 첫번째 Person의 복부 부분의 Joint를 Main body center로 저장(2번 Joint)
        '''
        이상한 게 첫번째 프레임의 2번 Joint 좌표를 중심으로 보는 것이 아니라
        모든 프레임에서 2번 Joint 좌표를 중심으로 바라봄.
        이 경우, 달리기는 제자리기 뛰기가 되는 단점이 존재할 것 같음.
        '''
        
        main_body_center = skeleton[0][:, 1:2, :].copy() # skeleton: (M, T, V, C) (2, 300, 25, 3) -> main body center: (T, 1, C)
        
        # 전체 스켈레톤에 대해
        for i_p, person in enumerate(skeleton): # Dimension M (# person)  # person : (T, V, C)
            
            # Person 데이터가 전부 0인 경우(Person 1과 Person 2가 전부 0)이면 넘기기(continue)
            if person.sum() == 0: # person : (T, V, C)  frames (T), nodes (V), channels (C)
                continue
                
            # For all `person`, compute the `mask` which is the non-zero channel dimension
            # Person 데이터가 존재하는 경우, mask는 X, Y, Z 값의 합이 0이 아니면(Action이 있으면) True, 0이면 False
            mask = (person.sum(-1) != 0).reshape(T, V, 1) 
            
            # Subtract the first skeleton's centre joint
            # Skeleton data의 2번째 Joint를 모두 0, 0, 0으로 이동시킴.
            s[i_s, i_p] = (s[i_s, i_p] - main_body_center) * mask

    # 척추 부분을 Z축과 Parallel하게 만들음.
    print('parallel the bone between hip(jpt 0) and spine(jpt 1) of the first person to the z axis')
    for i_s, skeleton in enumerate(tqdm(s)): # skeleton: (M, T, V, C) (2, 300, 25, 3) # persons (M), frames (T), nodes (V), channels (C) 
        # Person 데이터가 전부 0인 경우(Person 1과 Person 2가 전부 0)이면 넘기기
        if skeleton.sum() == 0:
            continue
        
        # Shapes: (C,)
        joint_bottom = skeleton[0, 0, zaxis[0]] # zxais = [0, 1]이면, 1번 Joint  
        joint_top = skeleton[0, 0, zaxis[1]] # zxais = [0, 1]이면, 2번 Joint  
        axis = np.cross(joint_top - joint_bottom, [0, 0, 1]) # bottom to top vector. XY평면이 지평면
        angle = angle_between(joint_top - joint_bottom, [0, 0, 1])
        matrix_z = rotation_matrix(axis, angle)
        for i_p, person in enumerate(skeleton):
            if person.sum() == 0:
                continue
            for i_f, frame in enumerate(person):
                if frame.sum() == 0:
                    continue
                for i_j, joint in enumerate(frame):
                    s[i_s, i_p, i_f, i_j] = np.dot(matrix_z, joint)

    # 양쪽 어깨선을 X축과 나란하게 만들음.
    print('parallel the bone between right shoulder(jpt 8) and left shoulder(jpt 4) of the first person to the x axis')
    for i_s, skeleton in enumerate(tqdm(s)):
        if skeleton.sum() == 0:
            continue
        joint_rshoulder = skeleton[0, 0, xaxis[0]] # xaxis = [8, 4]
        joint_lshoulder = skeleton[0, 0, xaxis[1]]
        axis = np.cross(joint_rshoulder - joint_lshoulder, [1, 0, 0])
        angle = angle_between(joint_rshoulder - joint_lshoulder, [1, 0, 0])
        matrix_x = rotation_matrix(axis, angle)
        for i_p, person in enumerate(skeleton):
            if person.sum() == 0:
                continue
            for i_f, frame in enumerate(person):
                if frame.sum() == 0:
                    continue
                for i_j, joint in enumerate(frame):
                    s[i_s, i_p, i_f, i_j] = np.dot(matrix_x, joint)

    data = np.transpose(s, [0, 4, 2, 3, 1])
    return data

if __name__ == '__main__':
    data = np.load('../data/ntu/xview/val_data.npy')
    pre_normalization(data)
    np.save('../data/ntu/xview/data_val_pre.npy', data)
