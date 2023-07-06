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

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F

PROP = 0.50
IMAGE_SIZE = 256
BATCH_SIZE = 32 // 2
#EPOCHS = 114
EPOCHS = 571
LR = 0.001
WEIGHT_DECAY = 2e-4


# LOG_INTERATION_INTERVAL = 100
# # TRAIN_FILE_COUNT = 4000
# TEST_INTERATION_INTERVAL = 1000
#STEP_SIZE = 114
TRAIN_FILE_COUNT = 14000
TRAIN_PRINT_FREQUENCY = 100
EVAL_PRINT_FREQUENCY = 1
#DECAY_EPOCH = [57]
DECAY_EPOCH = [457]
#DECAY_EPOCH = [80,140,180]
TMP = 10

OUTPUT_PATH = Path(__file__).stem
try:
  os.makedirs(OUTPUT_PATH)
except OSError:
  pass

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



class Layer_T1(nn.Module):
  def __init__(self, in_channel, out_channel):
    super(Layer_T1, self).__init__()

    self.in_channel = in_channel
    self.out_channel = out_channel

    self.layers = nn.Sequential(
      nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, bias=False),
      nn.BatchNorm2d(out_channel),
      nn.ReLU(),
    )


  def forward(self, input):
    output = self.layers(input)

    return output


class Layer_T2(nn.Module):
  def __init__(self, in_channel, out_channel):
    super(Layer_T2, self).__init__()

    self.in_channel = in_channel
    self.out_channel = out_channel

    self.layers = nn.Sequential(
      Layer_T1(in_channel, out_channel),
      nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, bias=False),
      nn.BatchNorm2d(out_channel),
    )


  def forward(self, input):
    output = self.layers(input)
    ouptut = output + input

    return output


class Layer_T3(nn.Module):
  def __init__(self, in_channel, out_channel):
    super(Layer_T3, self).__init__()

    self.in_channel = in_channel
    self.out_channel = out_channel

    self.layers = nn.Sequential(
      Layer_T1(in_channel, out_channel),
      nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, bias=False),
      nn.BatchNorm2d(out_channel),
      nn.AvgPool2d(kernel_size=3, padding=1, stride=2),
    )

    self.layers_residual = nn.Sequential(
      nn.Conv2d(in_channel, out_channel, kernel_size=1, bias=False, stride=2),
      nn.BatchNorm2d(out_channel),
    )

  def forward(self, input):
    output = self.layers(input)
    residual = self.layers_residual(input)

    output = output + residual

    return output


class Layer_T4(nn.Module):
  def __init__(self, in_channel, out_channel, pool_size):
    super(Layer_T4, self).__init__()

    self.in_channel = in_channel
    self.out_channel = out_channel

    self.layers = nn.Sequential(
      Layer_T1(in_channel, out_channel),
      nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, bias=False),
      nn.BatchNorm2d(out_channel),
      nn.AvgPool2d(kernel_size=pool_size, stride=1),
    )


  def forward(self, input):
    output = self.layers(input)

    return output


class SRNet(nn.Module):
  def __init__(self):
    super(SRNet, self).__init__()

    self.layer1 = Layer_T1(1, 64)
    self.layer2 = Layer_T1(64, 16)

    self.layer3 = Layer_T2(16, 16)
    self.layer4 = Layer_T2(16, 16)
    self.layer5 = Layer_T2(16, 16)
    self.layer6 = Layer_T2(16, 16)
    self.layer7 = Layer_T2(16, 16)

    self.layer8 = Layer_T3(16, 16)
    self.layer9 = Layer_T3(16, 64)
    self.layer10 = Layer_T3(64, 128)
    self.layer11 = Layer_T3(128, 256)

    self.layer12 = Layer_T4(256, 512, pool_size=16)

    self.fc1 = nn.Linear(1 * 1 * 512, 2)

  def forward(self, input):
    output = input

    output = self.layer1(output)
    output = self.layer2(output)

    output = self.layer3(output)
    output = self.layer4(output)
    output = self.layer5(output)
    output = self.layer6(output)
    output = self.layer7(output)

    output = self.layer8(output)
    output = self.layer9(output)
    output = self.layer10(output)
    output = self.layer11(output)

    output = self.layer12(output)


    output = output.view(output.size(0), -1)
    output = self.fc1(output)

    return output


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
  batch_time = AverageMeter()
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
    output = model(data)


    criterion = nn.CrossEntropyLoss()
    loss = criterion(output, label)

    losses.update(loss.item(), data.size(0))

    loss.backward()
    optimizer.step()

    batch_time.update(time.time() - end)
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


def evaluate(model, device, eval_loader, epoch, optimizer, best_acc,PARAMS_PATH, TMP): #  
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
  
  if accuracy>best_acc and epoch>TMP:
    best_acc=accuracy
    all_state = {
      'original_state':model.state_dict(),
      'optimizer_state':optimizer.state_dict(),
      'epoch':epoch
    }
    torch.save(all_state, PARAMS_PATH)

  logging.info('-' * 8)
  logging.info('Eval accuracy: {:.4f}'.format(accuracy))
  logging.info('Best accuracy:{:.4f}'.format(best_acc))    
  logging.info('-' * 8)    
  return best_acc
  


def initWeights(module):
  if type(module) == nn.Conv2d:
    if module.weight.requires_grad:
      nn.init.kaiming_normal_(module.weight.data, mode='fan_in', nonlinearity='relu')

      # nn.init.xavier_uniform_(module.weight.data)
      # nn.init.constant_(module.bias.data, val=0.2)
    # else:
    #   module.weight.requires_grad = True

  if type(module) == nn.Linear:
    nn.init.normal_(module.weight.data, mean=0, std=0.01)
    nn.init.constant_(module.bias.data, val=0)

  if type(module) == nn.BatchNorm2d:
    nn.init.constant_(module.weight.data, val=1)
    nn.init.constant_(module.bias.data, val=0)


class AugData():
  def __call__(self, sample):
    data, label = sample['data'], sample['label']

    rot = random.randint(0,3)

    #data = np.rot90(data, rot, axes=[2, 3]).copy()
    data = np.rot90(data, rot, axes=[1, 2]).copy()  


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
  def __init__(self, index_path, BOSSBASE_COVER_DIR, BOSSBASE_STEGO_DIR, BOWS_COVER_DIR, BOWS_STEGO_DIR,transform=None):
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

  #setLogger(LOG_PATH, mode='w')

  #Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)
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
  TIMES = args.times
  EPOCH1 = args.epoch1
  
  TRAIN_INDEX_PATH = '../index_list/bossbase_train_index.npy'
  VALID_INDEX_PATH = '../index_list/bossbase_valid_index.npy'
  TEST_INDEX_PATH = '../index_list/bossbase_test_index.npy'
  # 01 001 0001

  BOSSBASE_COVER_DIR = ''
  BOSSBASE_STEGO_DIR = ''

  BOWS_COVER_DIR = ''
  BOWS_STEGO_DIR = ''
  train_dataset = MyDataset(TRAIN_INDEX_PATH, BOSSBASE_COVER_DIR, BOSSBASE_STEGO_DIR, BOWS_COVER_DIR, BOWS_STEGO_DIR, train_transform)
  valid_dataset = MyDataset(VALID_INDEX_PATH, BOSSBASE_COVER_DIR, BOSSBASE_STEGO_DIR, BOWS_COVER_DIR, BOWS_STEGO_DIR, eval_transform)
  test_dataset = MyDataset(TEST_INDEX_PATH, BOSSBASE_COVER_DIR, BOSSBASE_STEGO_DIR, BOWS_COVER_DIR, BOWS_STEGO_DIR, eval_transform)


  train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, **kwargs)
  valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, **kwargs)
  test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, **kwargs)

  LOAD_RATE = float(EMBEDDING_RATE) + 0.1
  LOAD_RATE = round(LOAD_RATE, 1)

  PARAMS_NAME = 'params-{}-{}.pt'.format(STEGANOGRAPHY, EMBEDDING_RATE)
  LOG_NAME = 'model_log-{}-{}'.format(STEGANOGRAPHY, EMBEDDING_RATE)
  '''
  PARAMS_NAME = '{}-{}-{}-params-{}-lr={}-{}.pt'.format(STEGANOGRAPHY, EMBEDDING_RATE, DATASET_INDEX, TIMES, LR,EPOCH1)
  LOG_NAME = '{}-{}-{}-model_log-{}-lr={}-{}'.format(STEGANOGRAPHY, EMBEDDING_RATE, DATASET_INDEX, TIMES, LR,EPOCH1)
  '''
  PARAMS_PATH = os.path.join(OUTPUT_PATH, PARAMS_NAME)
  LOG_PATH = os.path.join(OUTPUT_PATH, LOG_NAME)
  
  #transfer learning 
  PARAMS_INIT_NAME = '{}-{}-{}-params-{}-lr={}-{}.pt'.format(STEGANOGRAPHY, LOAD_RATE, DATASET_INDEX, TIMES, LR, EPOCH1)
  
  if LOAD_RATE == 0.4:
    PARAMS_INIT_NAME = '{}-{}-{}-params-{}-lr={}-{}.pt'.format(STEGANOGRAPHY, LOAD_RATE, DATASET_INDEX, TIMES, '0.001', 571)
    
  if LOAD_RATE == 0.3:
  	PARAMS_INIT_NAME = '{}-{}-{}-params-{}-lr={}-{}.pt'.format(STEGANOGRAPHY, LOAD_RATE, DATASET_INDEX, TIMES, '0.001', 114)
   
  
  PARAMS_INIT_PATH = os.path.join(OUTPUT_PATH, PARAMS_INIT_NAME)
  
  setLogger(LOG_PATH, mode='w')

  Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)

  model = SRNet().to(device)
  model.apply(initWeights)

  params = model.parameters()

  params_wd, params_rest = [], []
  for param_item in params:
      if param_item.requires_grad:
          (params_wd if param_item.dim() != 1 else params_rest).append(param_item)

  param_groups = [{'params': params_wd, 'weight_decay': WEIGHT_DECAY},
                    {'params': params_rest}]

  optimizer = optim.Adamax(param_groups, lr=LR)
    
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
  
  if LOAD_RATE != 0.5:
    all_state = torch.load(PARAMS_INIT_PATH)
    original_state = all_state['original_state']
    model.load_state_dict(original_state)
  

  scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=DECAY_EPOCH, gamma=0.1)
  
  best_acc=0.0 

  train_hist = {}
  train_hist['acc'] = []
  train_hist['err'] = []

  for epoch in range(startEpoch, EPOCHS + 1):
    scheduler.step()

    train(model, device, train_loader, optimizer, epoch)

    if epoch % EVAL_PRINT_FREQUENCY == 0:
      adjust_bn_stats(model, device, train_loader)
      best_acc=evaluate(model, device, valid_loader, epoch, optimizer, best_acc, PARAMS_PATH, TMP) # 
      print('best_acc')
      print(best_acc)

      if best_acc > 0:
        train_hist['acc'].append(best_acc)
        train_hist['err'].append(1-best_acc)
        acc_plot(train_hist, 'srnet_test', model_name = STEGANOGRAPHY)

  logging.info('\nTest set accuracy: ')
  
  #load best parmater to test    
  all_state = torch.load(PARAMS_PATH)
  original_state = all_state['original_state']
  optimizer_state = all_state['optimizer_state']
  epoch = all_state['epoch']
  model.load_state_dict(original_state)
  optimizer.load_state_dict(optimizer_state)
  
  #best_acc=0.0 #add
  adjust_bn_stats(model, device, train_loader)
  
  test_acc=evaluate(model, device, test_loader, epoch, optimizer, best_acc, PARAMS_PATH, TMP) #  
  print('test_acc', test_acc)
  print('test_err', 1-test_acc)


def myParseArgs():
  parser = argparse.ArgumentParser()

  parser.add_argument(
    '-i',
    '--DATASET_INDEX',
    help='Path for loading dataset',
    type=str,
    default=''
  )

  parser.add_argument(
    '-alg',
    '--STEGANOGRAPHY',
    help='embedding_algorithm',
    type=str,
    choices=['HILL-CMDC','s-uniward','j-uniward'],
    #required=True
    default='BossBase-hill-stego-1'
  )

  parser.add_argument(
    '-rate',
    '--EMBEDDING_RATE',
    help='embedding_rate',
    type=str,
    choices=['0.2', '0.3', '0.4'],
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
    '-t',
    '--times',
    help='Determine which gpu to use',
    type=str,
    #required=True
    default=''
  )
  
  parser.add_argument(
    '-epoch1',
    '--epoch1',
    help='epoch1',
    type=str,
    default=''
  )

  parser.add_argument(
    '-l',
    '--statePath',
    help='Path for loading model state',
    type=str,
    default=''
    #default='../models/HILL_srnet_epoch_500.pt'
  )

  args = parser.parse_args()

  return args


if __name__ == '__main__':
  args = myParseArgs()

  os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuNum
  main(args)

