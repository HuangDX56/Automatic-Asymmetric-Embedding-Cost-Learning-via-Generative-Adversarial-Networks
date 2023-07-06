#!/usr/bin/env python3

import os
import argparse
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
import copy
import logging
import random
import scipy.io as sio
import matplotlib.pyplot as plt
import time
from glob import glob

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F

from srm_filter_kernel import all_normalized_hpf_list
from MPNCOV import *  # MPNCOV


PROP = 0.50
IMAGE_SIZE = 256
BATCH_SIZE = 32 // 2
num_levels = 3
EPOCHS = 400
LR = 0.005

WEIGHT_DECAY = 5e-4

TRAIN_PRINT_FREQUENCY = 100
EVAL_PRINT_FREQUENCY = 1
DECAY_EPOCH = [50, 150, 250]


OUTPUT_PATH = Path(__file__).stem

def acc_plot(hist, path = '', model_name = ''):
    x = range(len(hist['acc']))
    y1 = hist['acc']
    y2 = hist['err']

    plt.plot(x,y1,label='acc')
    plt.plot(x,y2,label='err')

    plt.xlabel('Iter')
    plt.ylabel('Acc')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    path = os.path.join(path, 'acc_' + model_name + '.png')

    plt.savefig(path)

    plt.close()



class TLU(nn.Module):
  def __init__(self, threshold):
    super(TLU, self).__init__()

    self.threshold = threshold

  def forward(self, input):
    output = torch.clamp(input, min=-self.threshold, max=self.threshold)

    return output

# https://gist.github.com/erogol/a324cc054a3cdc30e278461da9f1a05e
class SPPLayer(nn.Module):
    def __init__(self, num_levels):
        super(SPPLayer, self).__init__()

        self.num_levels = num_levels

    def forward(self, x):
        bs, c, h, w = x.size()
        pooling_layers = []
        for i in range(self.num_levels):
            kernel_size = h // (2 ** i)

            tensor = F.avg_pool2d(x, kernel_size=kernel_size,
                                  stride=kernel_size).view(bs, -1)
            pooling_layers.append(tensor)
        x = torch.cat(pooling_layers, dim=-1)
        return x


# absult value operation
class ABS(nn.Module):
    def __init__(self):
        super(ABS, self).__init__()

    def forward(self, input):
        output = torch.abs(input)
        return output


# add operation
class ADD(nn.Module):
    def __init__(self):
        super(ADD, self).__init__()

    def forward(self, input1, input2):
        output = torch.add(input1, input2)
        return output


# Pre-processing Module
class HPF(nn.Module):
    def __init__(self):
        super(HPF, self).__init__()

        # Load 30 SRM Filters
        all_hpf_list_5x5 = []

        for hpf_item in all_normalized_hpf_list:
            if hpf_item.shape[0] == 3:
                hpf_item = np.pad(hpf_item, pad_width=((1, 1), (1, 1)), mode='constant')

            all_hpf_list_5x5.append(hpf_item)

        hpf_weight = nn.Parameter(torch.Tensor(all_hpf_list_5x5).view(30, 1, 5, 5), requires_grad=False)

        self.hpf = nn.Conv2d(1, 30, kernel_size=5, padding=2, bias=False)
        self.hpf.weight = hpf_weight

        # Truncation, threshold = 3
        self.tlu = TLU(3.0)

    def forward(self, input):

        output = self.hpf(input)
        output = self.tlu(output)

        return output


class YedNet(torch.nn.Module):
    def __init__(self):
        super(YedNet, self).__init__()
        self.group1 = HPF()  # pre-processing Layer 1
        # self.conv0.weight = torch.nn.Parameter(srm)

        self.conv1 = torch.nn.Conv2d(in_channels=30, out_channels=30, kernel_size=5, stride=1,
                                     padding=2)  # Sepconv Block 1 Layer 2
        self.abs = ABS()
        self.bn1 = nn.BatchNorm2d(30)
        # Trunc T= 3
        self.tlu3 = TLU(3.0)
        self.conv2 = torch.nn.Conv2d(in_channels=30, out_channels=30, kernel_size=5, stride=1,
                                     padding=2)  # Sepconv Block 2 Layer 3
        self.bn2 = nn.BatchNorm2d(30)
        # Trunc T = 1
        self.tlu1 = TLU(1.0)
        self.pool = torch.nn.AvgPool2d(kernel_size=5, stride=2,
                                       padding=2)  # the same pool layer well be used to L3 and L4
        self.conv3 = torch.nn.Conv2d(in_channels=30, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(32)

        self.conv4 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)  # Layer 4
        self.bn4 = nn.BatchNorm2d(64)

        self.conv5 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)  # Layer 5
        self.bn5 = nn.BatchNorm2d(128)

        self.conv6 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)  # Layer 6
        self.bn6 = nn.BatchNorm2d(64)

        self.conv7 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)  # Layer 7
        self.bn7 = nn.BatchNorm2d(128)

        # self.spp_layer = SPPLayer(spp_level) # spp_level = 1 Global averge pooling

        self.fc1 = torch.nn.Linear(128, 256)
        self.fc2 = torch.nn.Linear(256, 1024)
        self.fc3 = torch.nn.Linear(1024, 2)

    def forward(self, x):
        x = self.group1(x)
        x = self.conv1(x)
        x = self.abs(x)
        # x =  F.relu(x)
        x = self.bn1(x)
        x = self.tlu3(x)

        x = self.bn2(self.conv2(x))
        x = self.tlu1(x)
        x = self.pool(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)

        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)

        x = F.relu(self.bn5(self.conv5(x)))
        

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(-1, 128)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return (x)


class AverageMeter(object):
  def __init__(self):
    self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count


def train(model, device, train_loader, optimizer, epoch):
  batch_time = AverageMeter() #ONE EPOCH TRAIN TIME
  data_time = AverageMeter()
  losses = AverageMeter()

  model.train()

  end = time.time()

  for i, sample in enumerate(train_loader):

    data_time.update(time.time() - end) 

    data, label = sample['data'], sample['label']

    shape = list(data.size())
    data = data.reshape(shape[0] * shape[1], *shape[2:])
    label = label.reshape(-1)


    data, label = data.to(device), label.to(device)

    optimizer.zero_grad()

    end = time.time()

    output = model(data) #FP


    criterion = nn.CrossEntropyLoss()
    loss = criterion(output, label)

    losses.update(loss.item(), data.size(0))

    loss.backward()      #BP
    optimizer.step()

    batch_time.update(time.time() - end) #BATCH TIME = BATCH BP+FP
    end = time.time()

    if i % TRAIN_PRINT_FREQUENCY == 0:
      # logging.info('Epoch: [{}][{}/{}] \t Loss {:.6f}'.format(epoch, i, len(train_loader), loss.item()))

      logging.info('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))

def adjust_bn_stats(model, device, train_loader):
  model.train()

  with torch.no_grad():
    for sample in train_loader:
      data, label = sample['data'], sample['label']

      shape = list(data.size())
      data = data.reshape(shape[0] * shape[1], *shape[2:])
      label = label.reshape(-1)

      data, label = data.to(device), label.to(device)

      output = model(data)


def evaluate(model, device, eval_loader, epoch, optimizer, best_acc, PARAMS_PATH):
  model.eval()

  test_loss = 0
  correct = 0

  with torch.no_grad():
    for sample in eval_loader:
      data, label = sample['data'], sample['label']

      shape = list(data.size())
      data = data.reshape(shape[0] * shape[1], *shape[2:])
      label = label.reshape(-1)

      data, label = data.to(device), label.to(device)

      output = model(data)
      pred = output.max(1, keepdim=True)[1]
      correct += pred.eq(label.view_as(pred)).sum().item()

  accuracy = correct / (len(eval_loader.dataset) * 2)

  if accuracy > best_acc and epoch > 10:
    best_acc = accuracy
    all_state = {
      'original_state': model.state_dict(),
      'optimizer_state': optimizer.state_dict(),
      'epoch': epoch
    }
    torch.save(all_state, PARAMS_PATH)
  
  logging.info('-' * 8)
  logging.info('Eval accuracy: {:.4f}'.format(accuracy))
  logging.info('Eval err: {:.4f}'.format(1-accuracy))
  logging.info('Best accuracy:{:.4f}'.format(best_acc))   
  logging.info('-' * 8)

  return accuracy


def initWeights(module):
  if type(module) == nn.Conv2d:
    if module.weight.requires_grad:
      nn.init.kaiming_normal_(module.weight.data, mode='fan_in', nonlinearity='relu')


  if type(module) == nn.Linear:
    nn.init.normal_(module.weight.data, mean=0, std=0.01)
    nn.init.constant_(module.bias.data, val=0)


class AugData():
  def __call__(self, sample):
    data, label = sample['data'], sample['label']

    rot = random.randint(0,3)


    data = np.rot90(data, rot, axes=[1, 2]).copy()
    #for i in range(0,rot):
    #  data = np.rollaxis(data,1,2).copy()


    if random.random() < 0.5:
      data = np.flip(data, axis=2).copy()

    new_sample = {'data': data, 'label': label}

    return new_sample


class ToTensor():
  def __call__(self, sample):
    data, label = sample['data'], sample['label']

    data = np.expand_dims(data, axis=1)
    data = data.astype(np.float32)
    # data = data / 255.0

    new_sample = {
      'data': torch.from_numpy(data),
      'label': torch.from_numpy(label).long(),
    }

    return new_sample


class MyDataset(Dataset):
  def __init__(self, index_path, BOSSBASE_COVER_DIR, BOSSBASE_STEGO_DIR, BOWS_COVER_DIR, BOWS_STEGO_DIR, transform=None):
    self.index_list = np.load(index_path)
    self.transform = transform

    self.bossbase_cover_path = BOSSBASE_COVER_DIR + '/{}.pgm'
    self.bossbase_stego_path = BOSSBASE_STEGO_DIR + '/{}.pgm'

    self.bows_cover_path = BOWS_COVER_DIR + '/{}.pgm'
    self.bows_stego_path = BOWS_STEGO_DIR + '/{}.pgm'

  def __len__(self):
    return self.index_list.shape[0]

  def __getitem__(self, idx):
    file_index = self.index_list[idx]

    if file_index <= 10000:
      cover_path = self.bossbase_cover_path.format(file_index)
      stego_path = self.bossbase_stego_path.format(file_index)
    else:
      cover_path = self.bows_cover_path.format(file_index - 10000)
      stego_path = self.bows_stego_path.format(file_index - 10000)

    
    cover_data = cv2.imread(cover_path, -1)
    stego_data = cv2.imread(stego_path, -1)
    '''
    cover_data = sio.loadmat(cover_path)['img_mat']
    stego_data = sio.loadmat(stego_path)['img_mat']
    '''
    data = np.stack([cover_data, stego_data])
    
    label = np.array([0, 1], dtype='int32')

    sample = {'data': data, 'label': label}

    if self.transform:
      sample = self.transform(sample)

    return sample


def setLogger(log_path, mode='a'):
  logger = logging.getLogger()
  logger.setLevel(logging.INFO)

  if not logger.handlers:
    # Logging to a file
    file_handler = logging.FileHandler(log_path, mode=mode)
    file_handler.setFormatter(logging.Formatter('%(asctime)s: %(message)s', '%Y-%m-%d %H:%M:%S'))
    logger.addHandler(file_handler)

    # Logging to console
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(stream_handler)


def main(args):

#  setLogger(LOG_PATH, mode='w')

#  Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)
  statePath = args.statePath

  device = torch.device("cuda")

  kwargs = {'num_workers': 1, 'pin_memory': True}

  train_transform = transforms.Compose([
    AugData(),
    ToTensor()
  ])

  eval_transform = transforms.Compose([
    ToTensor()
  ])

  DATASET_INDEX = args.DATASET_INDEX
  STEGANOGRAPHY = args.STEGANOGRAPHY
  EMBEDDING_RATE = args.EMBEDDING_RATE
  
  BOSSBASE_COVER_DIR = ''
  BOSSBASE_STEGO_DIR = ''
  
  BOWS_COVER_DIR = ''
  BOWS_STEGO_DIR = ''

  TRAIN_INDEX_PATH = '../index_list/bossbase_train_index.npy'
  VALID_INDEX_PATH = '../index_list/bossbase_valid_index.npy'
  TEST_INDEX_PATH = '../index_list/bossbase_test_index.npy'
  # 01 001 0001

  PARAMS_NAME = '{}-{}-{}-{:.2f}-params.pt'.format(STEGANOGRAPHY, EMBEDDING_RATE, DATASET_INDEX,PROP)
  LOG_NAME = '{}-{}-{}-{:.2f}-model_log'.format(STEGANOGRAPHY, EMBEDDING_RATE, DATASET_INDEX,PROP)
  
  PARAMS_PATH = os.path.join(OUTPUT_PATH, PARAMS_NAME)
  LOG_PATH = os.path.join(OUTPUT_PATH, LOG_NAME)

  setLogger(LOG_PATH, mode='w')

  Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)
  
  train_dataset = MyDataset(TRAIN_INDEX_PATH, BOSSBASE_COVER_DIR, BOSSBASE_STEGO_DIR, BOWS_COVER_DIR, BOWS_STEGO_DIR, train_transform)
  valid_dataset = MyDataset(VALID_INDEX_PATH, BOSSBASE_COVER_DIR, BOSSBASE_STEGO_DIR, BOWS_COVER_DIR, BOWS_STEGO_DIR, eval_transform)
  test_dataset = MyDataset(TEST_INDEX_PATH, BOSSBASE_COVER_DIR, BOSSBASE_STEGO_DIR, BOWS_COVER_DIR, BOWS_STEGO_DIR, eval_transform)

  train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, **kwargs)
  valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, **kwargs)
  test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, **kwargs)


  model = YedNet().to(device)
  model.apply(initWeights)


  params = model.parameters()


  params_wd, params_rest = [], []
  for param_item in params:
      if param_item.requires_grad:
          (params_wd if param_item.dim() != 1 else params_rest).append(param_item)

  param_groups = [{'params': params_wd, 'weight_decay': WEIGHT_DECAY},
                    {'params': params_rest}]

  optimizer = optim.SGD(param_groups, lr=LR, momentum=0.9)

  if statePath:
    logging.info('-' * 8)
    logging.info('Load state_dict in {}'.format(statePath))
    logging.info('Load stego in {}'.format(BOSSBASE_STEGO_DIR))
    logging.info('Load index in {}'.format(TEST_INDEX_PATH))
    logging.info('-' * 8)

    all_state = torch.load(statePath)

    original_state = all_state['original_state']
    optimizer_state = all_state['optimizer_state']
    epoch = all_state['epoch']

    model.load_state_dict(original_state)
    optimizer.load_state_dict(optimizer_state)

    startEpoch = epoch + 1

  else:
    startEpoch = 1

  scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=DECAY_EPOCH, gamma=0.2)
  best_acc = 0.0
  
  train_hist = {}
  train_hist['acc'] = []
  train_hist['err'] = []

  for epoch in range(startEpoch, EPOCHS + 1):
    scheduler.step()

    train(model, device, train_loader, optimizer, epoch)

    if epoch % EVAL_PRINT_FREQUENCY == 0:
      adjust_bn_stats(model, device, train_loader)
      best_acc = evaluate(model, device, valid_loader, epoch, optimizer, best_acc, PARAMS_PATH) #
      
      train_hist['acc'].append(best_acc)
      train_hist['err'].append(1-best_acc)
      acc_plot(train_hist, 'yednet_test', model_name = STEGANOGRAPHY)

  logging.info('\nTest set accuracy: ')

   #load best parmater to test    
  all_state = torch.load(PARAMS_PATH)
  original_state = all_state['original_state']
  optimizer_state = all_state['optimizer_state']
  model.load_state_dict(original_state)
  optimizer.load_state_dict(optimizer_state)
  
  adjust_bn_stats(model, device, train_loader)
  
  test_acc=evaluate(model, device, test_loader, epoch, optimizer, best_acc, PARAMS_PATH) #
  print('test_acc', test_acc)
  print('test_err', 1-test_acc)

def myParseArgs():
  parser = argparse.ArgumentParser()

  parser.add_argument(
    '-i',
    '--DATASET_INDEX',
    help='Path for loading dataset',
    type=str,
    default='1'
  )

  parser.add_argument(
    '-alg',
    '--STEGANOGRAPHY',
    help='embedding_algorithm',
    type=str,
    default='xunet_0.001_SG'
  )

  parser.add_argument(
    '-rate',
    '--EMBEDDING_RATE',
    help='embedding_rate',
    type=str,
    choices=['0.1', '0.2', '0.3', '0.4'],
    #required=True
    default='0.4'
  )

  parser.add_argument(
    '-g',
    '--gpuNum',
    help='Determine which gpu to use',
    type=str,
    choices=['0', '1', '2', '3'],
    required=True
  )

  parser.add_argument(
    '-l',
    '--statePath',
    help='Path for loading model state',
    type=str,
    default=''
  )

  args = parser.parse_args()

  return args


if __name__ == '__main__':
  args = myParseArgs()

  os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuNum
  main(args)


