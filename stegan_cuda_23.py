import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
from torch.utils import data
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import argparse
import random
import numpy as np
import matplotlib.pyplot as plt
import os


from PIL import Image
import time
import cv2
import scipy.io as sio
from pathlib import Path

from ResUNet1DB import UNet
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

def myParseArgs():
    parser = argparse.ArgumentParser()

    parser.add_argument('--netG', default='', help="path to netG (to continue training)")

    parser.add_argument('--config', default='', help="train result")

    parser.add_argument('--datacover', help='path to dataset',default='')
    parser.add_argument('--indexpath', help='path to index',default='./index_list/boss_gan_train_1w.npy')

    parser.add_argument('--train_cover', help='path to dataset',default='')
    parser.add_argument('--train_indexpath', help='path to index',default='./index_list/szu_gan_train_4w.npy')
    
    parser.add_argument('--root', default='', help="path to save result")

    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--batchSize', type=int, default=50, help='input batch size')
    parser.add_argument('--imageSize', type=int, default=256, help='the height / width of the input image to network')
    
    parser.add_argument('--TS_PARA', type=int, default=1000000, help='parameter for double tanh')
    parser.add_argument('--payload', type=float, default=0.4, help='embeding rate')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    
    args = parser.parse_args()
    return args



class Dataset(data.Dataset):
    def __init__(self, cover_path, index_path, transforms = None):
        self.index_list = np.load(index_path)
        self.cover_path = cover_path+'{}.pgm'

        self.transforms = transforms        
        

    def __getitem__(self,index):
        file_index = self.index_list[index]
        cover_path = self.cover_path.format(file_index)
        cover = cv2.imread(cover_path, -1)
        label = np.array([0, 1], dtype='int32')
        
        rand_ud = np.random.rand(256, 256)
        sample = {'cover':cover, 'rand_ud':rand_ud, 'label':label, 'index':file_index}
        
        if self.transforms:
            sample = self.transforms(sample)
            
        return sample

        
    def __len__(self):
        return len(self.index_list)


class ToTensor():
  def __call__(self, sample):
    cover, rand_ud, label, index  = sample['cover'], sample['rand_ud'], sample['label'], sample['index']

    cover = np.expand_dims(cover, axis=0)
    cover = cover.astype(np.float32)
    
    rand_ud = rand_ud.astype(np.float32)
    rand_ud = np.expand_dims(rand_ud,axis = 0)
    

    new_sample = {
      'cover': torch.from_numpy(cover),
      'rand_ud': torch.from_numpy(rand_ud),
      'label': torch.from_numpy(label).long(),
      'index':index
    }

    return new_sample


# custom weights initialization called on netG and netD
def weights_init_g(net):
    for m in net.modules():
        if isinstance(m,nn.Conv2d) and m.weight.requires_grad:
            m.weight.data.normal_(0., 0.02)
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0., 0.02)
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0., 0.02)
            m.bias.data.fill_(0)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)


def adjust_bn_stats(model, train_loader):
  model.train()

  with torch.no_grad():
    for i,sample in enumerate(train_loader,0):

        cover, rand_ud, label, index= sample['cover'], sample['rand_ud'], sample['label'], sample['index']
        cover, n, label = cover.cuda(), rand_ud.cuda(), label.cuda()

        # learn the probas
        p = model(cover)


def main():
    
    args = myParseArgs()
    try:
        stego_path = os.path.join(args.root, args.config)
        os.makedirs(stego_path)
    except OSError:
        pass

    if args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", args.manualSeed)
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)

    cudnn.benchmark = True
        
    
    transform = transforms.Compose([ToTensor(),])
    dataset = Dataset(args.datacover, args.indexpath, transforms=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batchSize,
                                            shuffle=True, num_workers=int(args.workers),drop_last = True)


    train_dataset = Dataset(args.train_cover, args.train_indexpath, transforms=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchSize,
                                            shuffle=True, num_workers=int(args.workers),drop_last = True)

    netG = UNet()
    netG = nn.DataParallel(netG)
    netG = netG.cuda()
    #netG.apply(weights_init_g)
    
    if args.netG != '':
        print('Load netG state_dict in {}'.format(args.netG))
        netG.load_state_dict(torch.load(args.netG))
    

    adjust_bn_stats(netG, train_loader)

    netG.eval()
    with torch.no_grad():
        for i,sample in enumerate(dataloader,0):

            cover, rand_ud, label, index= sample['cover'], sample['rand_ud'], sample['label'], sample['index']
            cover, n, label = cover.cuda(), rand_ud.cuda(), label.cuda()

            # learn the probas
            
            p_p, p_m = netG(cover)

            
            p_plus = p_p/2.0 + 1e-5
            p_minus = p_m/2.0 + 1e-5
            
            rho_plus = torch.log(1/p_plus-2)
            rho_minus = torch.log(1/p_minus-2)
            
            m = torch.zeros_like(cover)
            m[n < p_plus] = 1
            m[n > (torch.ones_like(cover)-p_minus)] = -1

            stego = cover + m

            stego[stego == -1] = 0
            stego[stego == 256] = 255

            # save images
            for k in range(0,stego.shape[0]):
                cost_p1 = rho_plus[k,0].detach().cpu().numpy()
                cost_p1[np.isinf(cost_p1)] = 10e+10
                cost_m1 = rho_minus[k,0].detach().cpu().numpy()
                cost_m1[np.isinf(cost_m1)] = 10e+10
                sio.savemat( ('%s%s/%d_cost.mat'%(args.root, args.config, index[k])), mdict={'cost_p1': cost_p1, 'cost_m1': cost_m1})


                img = stego[k,].detach().cpu().permute(1,2,0).type(torch.uint8).numpy()
                #cv2.imwrite('%s%s/%d_stego.png' %(args.root, args.config, index[k]), img)

                modify = (torch.round(m)+2)%2*255
                modify = modify[k,].detach().cpu().permute(1,2,0).type(torch.uint8).numpy()
                #cv2.imwrite('%s%s/%d_modify.png' %(args.root, args.config, index[k]), modify)

                pro_m1 = p_minus*255
                probas_m1 = pro_m1[k,].detach().cpu().permute(1,2,0).type(torch.uint8).numpy()
                #cv2.imwrite('%s%s/%d_probas.png' %(args.root,  args.config, index[k]), probas_m1)

                pro_p1 = p_plus*255
                probas_p1 = pro_p1[k,].detach().cpu().permute(1,2,0).type(torch.uint8).numpy()
                #cv2.imwrite('%s%s/%d_probas.png' %(args.root,  args.config, index[k]), probas_p1)
                
                im = Image.fromarray(np.uint8(img[:,:,0]))
                im.save( ('%s%s/%d.pgm' %(args.root, args.config, index[k])))


                mod = Image.fromarray(np.uint8(modify[:,:,0]))
                p_m1 = Image.fromarray(np.uint8(probas_m1[:,:,0]))
                p_p1 = Image.fromarray(np.uint8(probas_p1[:,:,0]))

                imslst = [mod, p_p1, p_m1, im]
                result = Image.new('RGBA', (args.imageSize * len(imslst), args.imageSize))
                for i, im in enumerate(imslst):
                    result.paste(im, box=(i * args.imageSize , 0))
                result.save(('%s%s/%d.png' %(args.root, args.config, index[k])))


    print('Stegan down for {}'.format(args.datacover))
    print('Output path {}'.format(args.root + args.config))
            

if __name__ == '__main__':
    main()