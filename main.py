#!/usr/bin/env python
from __future__ import print_function
import argparse
import os
import time
import numpy as np
import yaml
import pickle
from collections import OrderedDict, defaultdict
# torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
from torch.optim.lr_scheduler import MultiStepLR
import random
import inspect
import torch.backends.cudnn as cudnn

def init_seed(_):
    torch.cuda.manual_seed_all(1)
    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_parser():
    # parameter priority: command line > config > default
    parser = argparse.ArgumentParser(
        description='Directed Graph Neural Net for Skeleton Action Recognition')
    
    parser.add_argument(
        '--work-dir',
        default='./work_dir/temp',
        help='the work folder for storing results') # work_dir. 추가로 주석 달기

    parser.add_argument(
        '--model-saved-name', default='') # 추가로 주석 달기
    parser.add_argument(
        '--config',
        default='./config/nturgbd-cross-subject/train_spatial.yaml',
        help='path to the configuration file') # config 파일이 위치하는 디렉토리 

    # processor
    parser.add_argument(
        '--phase', default='train', help='must be train or test') # Train할 지, Test할 지 정하는 인자.
    parser.add_argument(
        '--save-score',
        type=str2bool,
        default=False,
        help='if ture, the classification score will be stored') # 추가로 주석 달기

    # visulize and debug
    parser.add_argument(
        '--seed', type=int, default=1, help='random seed for pytorch') # random seed 
    parser.add_argument(
        '--log-interval',
        type=int,
        default=100,
        help='the interval for printing messages (#iteration)') # log massage 출력할 Interval
    parser.add_argument(
        '--save-interval',
        type=int,
        default=2,
        help='the interval for storing models (#iteration)') # model을 저장할 Interval 
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=5,
        help='the interval for evaluating models (#iteration)') # Evaluate할 Interval
    parser.add_argument(
        '--print-log',
        type=str2bool,
        default=True,
        help='print logging or not') # Log 출력할지에 대한 여부
    parser.add_argument(
        '--show-topk',
        type=int,
        default=[1, 2],
        nargs='+',
        help='which Top K accuracy will be shown') # 추후에 주석 달기

    # feeder
    parser.add_argument(
        '--feeder', default='feeder.feeder', help='data loader will be used') # 추후에 주석 달기
    parser.add_argument(
        '--num-worker',
        type=int,
        default=os.cpu_count(),
        help='the number of worker for data loader') # 추후에 주석 달기
    parser.add_argument(
        '--train-feeder-args',
        default=dict(),
        help='the arguments of data loader for training') # 추후에 주석 달기
    parser.add_argument(
        '--test-feeder-args',
        default=dict(),
        help='the arguments of data loader for test') # 추후에 주석 달기

    # model
    parser.add_argument(
        '--model', default=None, help='the model will be used') # 사용할 Model
    parser.add_argument(
        '--model-args',
        type=dict,
        default=dict(),
        help='the arguments of model') # 사용할 Model의 argument # 추후에 주석 추가
    parser.add_argument(
        '--weights',
        default=None,
        help='the weights for network initialization') # Initialize할 weights
    parser.add_argument(
        '--ignore-weights',
        type=str,
        default=[],
        nargs='+',
        help='the name of weights which will be ignored in the initialization') # 추후에 주석 달기.

    # optim
    parser.add_argument(
        '--base-lr', type=float, default=0.01, help='initial learning rate') # 초기 Learning rate
    parser.add_argument(
        '--step',
        type=int,
        default=[60, 90],
        nargs='+',
        help='the epoch where optimizer reduce the learning rate') # Learning rate를 감소시킬 Epoch
    parser.add_argument(
        '--device',
        type=int,
        default=0,
        nargs='+',
        help='the indexes of GPUs for training or testing') # 사용할 GPU 번호. 여러개 입력 가능.
    parser.add_argument(
        '--optimizer', default='SGD', help='type of optimizer') # Optimizer 종류
    parser.add_argument(
        '--nesterov', type=str2bool, default=True, help='use nesterov or not') # Nesterov 사용 여부
    parser.add_argument(
        '--batch-size', type=int, default=32, help='training batch size') # Training시 Batch 사이즈
    parser.add_argument(
        '--test-batch-size', type=int, default=32, help='test batch size') # Test시 Batch 사이즈
    parser.add_argument(
        '--start-epoch',
        type=int,
        default=0,
        help='start training from which epoch') # Training 시 시작할 Epoch
    parser.add_argument(
        '--num-epoch',
        type=int,
        default=120,
        help='stop training in which epoch') # Epoch 크기
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=0.0001,
        help='weight decay for optimizer') # weight decay
    parser.add_argument(
        '--freeze-graph-until',
        type=int,
        default=10,
        help='number of epochs before making graphs learnable') # Graph Freeze를 할 Epoch

    # parser.add_argument('--only_train_part', default=False)
    # parser.add_argument('--only_train_epoch', default=0)
    # parser.add_argument('--warm_up_epoch', default=0)
    return parser


class Processor():
    """Processor for Skeleton-based Action Recognition"""
    def __init__(self, arg):
        self.arg = arg
        
        # work_dir에 config 파일 생성 및 저장
        self.save_arg()
        
        # phase가 Train일 때 
        if arg.phase == 'train':
            # train_feeder_args['debug']가 False 이고 
            if not arg.train_feeder_args['debug']:
                # 모델 파라미터를 저장할 model_saved_name 디렉토리가 존재하면
                if os.path.isdir(arg.model_saved_name):
                    print('log_dir: ', arg.model_saved_name, 'already exist')
                    answer = input('delete it? [y]/n:')
                    #  삭제를 선택하면 
                    if answer.lower() in ('y', ''):
                        shutil.rmtree(arg.model_saved_name) # 지정된 디렉토리의 모든 파일 삭제
                        print('Dir removed: ', arg.model_saved_name) 
                        input('Refresh the website of tensorboard by pressing any keys')
                    # 삭제하지 않을거면
                    else:
                        print('Dir not removed: ', arg.model_saved_name)

            self.train_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'train'), 'train')
            self.val_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'val'), 'val')
            # self.writer = SummaryWriter(os.path.join(arg.model_saved_name, 'training'), 'both')

        # num_point 수 데이터에서 가져오기
        if self.arg.phase == 'train': 
            joint_data = np.load(self.arg.train_feeder_args['joint_data_path'])
        elif self.arg.phase == 'test':
            joint_data = np.load(self.arg.test_feeder_args['joint_data_path'])
        
        self.arg.model_args['num_point'] = joint_data.shape[3]
        del joint_data
            
        self.global_step = 0
        self.load_model() # Model 선언 / Parameter Load / GPU 설정        
        self.load_param_groups() # Group parameters to apply different learning rules # Parameter 그룹 분할
        self.load_optimizer() # Optimizer 설정
        self.load_data() # data load
        self.lr = self.arg.base_lr
        self.best_acc = 0
        self.best_acc_epoch = 0

    # data load
    def load_data(self):
        Feeder = import_class(self.arg.feeder) # self.arg.feeder : feeders.feeder.Feeder
        self.data_loader = dict()
        if self.arg.phase == 'train':
            self.data_loader['train'] = torch.utils.data.DataLoader(dataset=Feeder(**self.arg.train_feeder_args),
                                                                    batch_size=self.arg.batch_size,
                                                                    shuffle=True,
                                                                    num_workers=self.arg.num_worker,
                                                                    drop_last=True,
                                                                    worker_init_fn=init_seed)

        # Load test data regardless
        self.data_loader['test'] = torch.utils.data.DataLoader(dataset=Feeder(**self.arg.test_feeder_args),
                                                               batch_size=self.arg.test_batch_size,
                                                               shuffle=False,
                                                               num_workers=self.arg.num_worker,
                                                               drop_last=False,
                                                               worker_init_fn=init_seed)
    
    # Model 선언 / Parameter Load / GPU 설정
    def load_model(self):
        # 출력으로 사용할 device
        output_device = self.arg.device[0] if type(self.arg.device) is list else self.arg.device        
        self.output_device = output_device
        Model = import_class(self.arg.model) # In here, self.arg.model = model.dgnn.Model
        
        # Copy model file to output dir
        # Model Class 파일을 work_dir에 복사        
        shutil.copy2(inspect.getfile(Model), self.arg.work_dir) # inspect.getfile(Model): ./model/dgnn.py

        # argument 값을 이용해서 model 객체를 생성하고 model과 loss를 device로 이동
        # self.arg.model_args['graph']: graph.directed_ntu_rgb_d.Graph
        # 'num_class': 10, 'num_point': 25, 'num_person': 2, 'graph': 'graph.directed_ntu_rgb_d.Graph'
        self.model = Model(**self.arg.model_args).cuda(output_device) 
        self.loss = nn.CrossEntropyLoss().cuda(output_device)

        # Load weights
        # 저장해두었던 모델의 weights가 있으면 weight load.
        # scratch train할 때는 코드가 작동하지 않음.
        if self.arg.weights:
            self.global_step = int(arg.weights[:-3].split('-')[-1])
            self.print_log('Load weights from {}.'.format(self.arg.weights))
            if '.pkl' in self.arg.weights:
                with open(self.arg.weights, 'r') as f:
                    weights = pickle.load(f)
            else:
                weights = torch.load(self.arg.weights)

            weights = OrderedDict(
                [[k.split('module.')[-1],
                  v.cuda(output_device)] for k, v in weights.items()])

            for w in self.arg.ignore_weights:
                if weights.pop(w, None) is not None:
                    self.print_log('Sucessfully Remove Weights: {}.'.format(w))
                else:
                    self.print_log('Can Not Remove Weights: {}.'.format(w))

            try:
                self.model.load_state_dict(weights)
            except:
                state = self.model.state_dict()
                diff = list(set(state.keys()).difference(set(weights.keys())))
                print('Can not find these weights:')
                for d in diff:
                    print('  ' + d)
                state.update(weights)
                self.model.load_state_dict(state)

        # Parallelise data if mulitple GPUs
        # GPU가 여러개 있으면 Parallelise data 
        if type(self.arg.device) is list: # config GPU가 List 값으로 되어있고
            if len(self.arg.device) > 1: # 길이가 1보다 크면 
                # 모델이 DataParallel을 사용하게 설정
                self.model = nn.DataParallel(
                    self.model,
                    device_ids=self.arg.device,
                    output_device=output_device)


    # Optimizer 설정
    def load_optimizer(self):
        p_groups = list(self.optim_param_groups.values())
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                p_groups,
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                p_groups,
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError('Unsupported optimizer: {}'.format(self.arg.optimizer))

        self.lr_scheduler = MultiStepLR(self.optimizer, milestones=self.arg.step, gamma=0.1)

    # self.arg를 work_dir에 config 파일로 저장 
    def save_arg(self):
        # save arg
        arg_dict = vars(self.arg)
        
        # work_dir 없으면 폴더 생성
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
            
        # work_dir에 config 파일 생성 및 저장
        with open('{}/config.yaml'.format(self.arg.work_dir), 'w') as f:
            yaml.dump(arg_dict, f)

    def print_time(self):
        localtime = time.asctime(time.localtime(time.time()))
        self.print_log("Local current time :  " + localtime)

    def print_log(self, s, print_time=True):
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            s = '[ {} ] {}'.format(localtime, s)
        print(s)
        if self.arg.print_log:
            with open('{}/log.txt'.format(self.arg.work_dir), 'a') as f:
                print(s, file=f)

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

    # Parameter 그룹 분할
    def load_param_groups(self):
        self.param_groups = defaultdict(list) # list 형태의 아무것도 존재하지 않는 dictionary 생성
        for name, params in self.model.named_parameters():         
            # parameter가 Adaptive Graph인지, 그 이외것인지 구분
            if ('source_M' in name) or ('target_M' in name):
                self.param_groups['graph'].append(params)
            else:
                self.param_groups['other'].append(params)

        # NOTE: Different parameter groups should have different learning behaviour
        self.optim_param_groups = {
            'graph': {'params': self.param_groups['graph']},
            'other': {'params': self.param_groups['other']}
        }

    def update_graph_freeze(self, epoch):
        graph_requires_grad = (epoch > self.arg.freeze_graph_until)
        self.print_log('Graphs are {} at epoch {}'.format('learnable' if graph_requires_grad else 'frozen', epoch + 1))
        for param in self.param_groups['graph']:
            param.requires_grad = graph_requires_grad
        # graph_weight_decay = 0 if freeze_graphs else self.arg.weight_decay
        # NOTE: will decide later whether we need to change weight decay as well
        # self.optim_param_groups['graph']['weight_decay'] = graph_weight_decay

    def train(self, epoch, save_model=False):
        self.print_log('Training epoch: {}'.format(epoch + 1))
        self.model.train()
        
        loader = self.data_loader['train']
        loss_values = []
        self.train_writer.add_scalar('epoch', epoch, self.global_step)
        self.record_time()
        timer = dict(dataloader=0.001, model=0.001, statistics=0.001)

        # if self.arg.only_train_part:
        #     if epoch > self.arg.only_train_epoch:
        #         print('only train part, require grad')
        #         for key, value in self.model.named_parameters():
        #             if 'PA' in key:
        #                 value.requires_grad = True
        #     else:
        #         print('only train part, do not require grad')
        #         for key, value in self.model.named_parameters():
        #             if 'PA' in key:
        #                 value.requires_grad = False

        self.update_graph_freeze(epoch)

        process = tqdm(loader)
        # for batch_idx, (data, label, index) in enumerate(process):
        for batch_idx, (joint_data, bone_data, label, index) in enumerate(process):
            self.global_step += 1
            # get data
            with torch.no_grad():
                joint_data = joint_data.float().cuda(self.output_device)
                bone_data = bone_data.float().cuda(self.output_device)
                label = label.long().cuda(self.output_device)
                
            timer['dataloader'] += self.split_time()

            # Clear gradients
            self.optimizer.zero_grad()

            ################################
            # Multiple forward passes + 1 backward pass to simulate larger batch size
            real_batch_size = 16
            splits = len(joint_data) // real_batch_size
            assert len(joint_data) % real_batch_size == 0, 'Real batch size should be a factor of arg.batch_size!'

            for i in range(splits):
                left = i * real_batch_size
                right = left + real_batch_size
                batch_joint_data, batch_bone_data = joint_data[left:right], bone_data[left:right]
                batch_label = label[left:right]

                # forward
                output = self.model(batch_joint_data, batch_bone_data)
                if isinstance(output, tuple):
                    output, l1 = output
                    l1 = l1.mean()
                else:
                    l1 = 0

                loss = self.loss(output, batch_label) / float(splits)
                loss.backward()

                loss_values.append(loss.item())
                timer['model'] += self.split_time()

                # Display loss
                process.set_description('loss: {:.4f}'.format(loss.item()))

                value, predict_label = torch.max(output, 1)
                acc = torch.mean((predict_label == batch_label).float())
                self.train_writer.add_scalar('acc', acc, self.global_step)
                self.train_writer.add_scalar('loss', loss.item(), self.global_step)
                self.train_writer.add_scalar('loss_l1', l1, self.global_step)

            # Step after looping over batch splits
            self.optimizer.step()

            ###############################

            # # forward
            # output = self.model(joint_data, bone_data)
            # if isinstance(output, tuple):
            #     output, l1 = output
            #     l1 = l1.mean()
            # else:
            #     l1 = 0
            # loss = self.loss(output, label) + l1

            # # backward
            # loss.backward()
            # self.optimizer.step()

            ################################

            # loss_values.append(loss.item())
            # timer['model'] += self.split_time()

            # # Display loss
            # process.set_description('loss: {:.4f}'.format(loss.item()))

            # value, predict_label = torch.max(output, 1)
            # acc = torch.mean((predict_label == label).float())
            # self.writer.add_scalar('train/acc', acc, self.global_step)
            # self.writer.add_scalar('train/loss', loss.item(), self.global_step)
            # self.writer.add_scalar('train/loss_l1', l1, self.global_step)

            # statistics
            self.lr = self.optimizer.param_groups[0]['lr']
            self.train_writer.add_scalar('lr', self.lr, self.global_step)
            timer['statistics'] += self.split_time()

        # statistics of time consumption and loss
        proportion = {
            k: '{: 2d}%'.format(int(round(v * 100 / sum(timer.values()))))
            for k, v in timer.items()
        }
        self.print_log('\tMean training loss: {:.4f}.'.format(np.mean(loss_values)))
        self.print_log('\tTime consumption: [Data]{dataloader}, [Network]{model}'.format(**proportion))

        self.lr_scheduler.step(epoch)

        if save_model:
            state_dict = self.model.state_dict()
            weights = OrderedDict([[k.split('module.')[-1], v.cpu()] for k, v in state_dict.items()])
            torch.save(weights, self.arg.model_saved_name + '-' + str(epoch) + '-' + str(int(self.global_step)) + '.pt')

    def eval(self, epoch, save_score=False, loader_name=['test'], wrong_file=None, result_file=None):
        if wrong_file is not None:
            f_w = open(wrong_file, 'w')
        if result_file is not None:
            f_r = open(result_file, 'w')

        self.model.eval()
        self.print_log('Eval epoch: {}'.format(epoch + 1))

        for ln in loader_name:
            loss_values, score_batches = [], []
            step = 0
            process = tqdm(self.data_loader[ln])
            for batch_idx, (joint_data, bone_data, label, index) in enumerate(process):
                step += 1
                with torch.no_grad():
                    joint_data = joint_data.float().cuda(self.output_device)
                    bone_data = bone_data.float().cuda(self.output_device)
                    label = label.long().cuda(self.output_device)
                    output = self.model(joint_data, bone_data)
                    if isinstance(output, tuple):
                        output, l1 = output
                        l1 = l1.mean()
                    else:
                        l1 = 0

                    loss = self.loss(output, label)
                    score_batches.append(output.cpu().numpy())
                    loss_values.append(loss.item())
                    # Argmax over logits = labels
                    _, predict_label = torch.max(output, dim=1)

                    if wrong_file is not None or result_file is not None:
                        predict = list(predict_label.cpu().numpy())
                        true = list(label.cpu().numpy())
                        for i, pred in enumerate(predict):
                            if result_file is not None:
                                f_r.write('{},{}\n'.format(pred, true[i]))
                            if pred != true[i] and wrong_file is not None:
                                f_w.write('{},{},{}\n'.format(index[i], pred, true[i]))

            # Concatenate along the batch dimension, and 1st dim ~= `len(dataset)`
            score = np.concatenate(score_batches)
            loss = np.mean(loss_values)
            accuracy = self.data_loader[ln].dataset.top_k(score, 1)
            if accuracy > self.best_acc:
                self.best_acc = accuracy
                self.best_acc_epoch = epoch

            print('Accuracy: ', accuracy, ' Model: ', self.arg.model_saved_name)
            if self.arg.phase == 'train':
                self.val_writer.add_scalar('loss', loss, self.global_step)
                self.val_writer.add_scalar('loss_l1', l1, self.global_step)
                self.val_writer.add_scalar('acc', accuracy, self.global_step)

            self.print_log('\tMean {} loss of {} batches: {}.'.format(
                ln, len(self.data_loader[ln]), np.mean(loss_values)))

            for k in self.arg.show_topk:
                self.print_log('\tTop{}: {:.2f}%'.format(
                    k, 100 * self.data_loader[ln].dataset.top_k(score, k)))

            if save_score:
                score_dict = dict(zip(self.data_loader[ln].dataset.sample_name, score))
                with open('{}/epoch{}_{}_score.pkl'.format(
                        self.arg.work_dir, epoch + 1, ln), 'wb') as f:
                    pickle.dump(score_dict, f)

    def start(self):
        # phase가 Train일 때 
        if self.arg.phase == 'train':
            self.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))
            self.global_step = self.arg.start_epoch * len(self.data_loader['train']) / self.arg.batch_size
            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                if self.lr < 1e-3:
                    break
                save_model = ((epoch + 1) % self.arg.save_interval == 0) or (
                        epoch + 1 == self.arg.num_epoch)

                self.train(epoch, save_model=save_model)
                self.eval(epoch, save_score=self.arg.save_score, loader_name=['test'])

            print('Best accuracy: {}, epoch: {}, model_name: {}'
                  .format(self.best_acc, self.best_acc_epoch, self.arg.model_saved_name))

        # phase가 Test일 때 
        elif self.arg.phase == 'test':
            if not self.arg.test_feeder_args['debug']:
                wf = self.arg.model_saved_name + '_wrong.txt'
                rf = self.arg.model_saved_name + '_right.txt'
            else:
                wf = rf = None
            if self.arg.weights is None:
                raise ValueError('Please appoint --weights.')
            self.arg.print_log = False
            self.print_log('Model:   {}.'.format(self.arg.model))
            self.print_log('Weights: {}.'.format(self.arg.weights))
            self.eval(epoch=0, save_score=self.arg.save_score, loader_name=['test'], wrong_file=wf, result_file=rf)
            self.print_log('Done.\n')


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])  # import return model
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


if __name__ == '__main__':
    parser = get_parser()

    # load arg from config file
    p = parser.parse_args()

    p.config = './config/nturgbd-cross-subject/train_spatial.yaml'

    if p.config is not None:
        with open(p.config, 'r') as f:
            default_arg = yaml.load(f) # config 파일에 들어있는 keys. config key로 명명
        
        # 예외 처리
        key = vars(p).keys() # parser에 들어있는 key값. default key로 명명 
        for k in default_arg.keys():
            if k not in key: # config key가 default key에 들어있지 않으면 
                print('WRONG ARG: {}'.format(k)) # default key에 해당 키가 없음을 알림.
                assert (k in key) 
                
        parser.set_defaults(**default_arg) # 임의의 개수의 Keyword arguments를 받아서 default key -> config key로 변경

    arg = parser.parse_args() # config key가 반영된 argument
    init_seed(0) # 시드 초기화 
    processor = Processor(arg) #
    processor.start()
