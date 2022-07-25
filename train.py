import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torchvision
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tqdm import tqdm
from utils import *
from condgenerators import ConGeneratorResnet
from get_class import get_classes

parser = argparse.ArgumentParser(description='Conditional Adversarial Generator')
parser.add_argument('--train_dir', default='imagenet', help='imagenet')
parser.add_argument('--batch_size', type=int, default=12, help='Number of trainig samples/batch')
parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
parser.add_argument('--lr', type=float, default=2e-4, help='Initial learning rate')
parser.add_argument('--eps', type=int, default=16, help='Perturbation Budget')
parser.add_argument('--model_type', type=str, default='incv3', help='incv3, res152')
parser.add_argument('--start-epoch', type=int, default=0, help='start-epoch')
parser.add_argument('--print_iters', type=int, default=10, help='print')
parser.add_argument('--method', type=str, default='cond', help='method')
parser.add_argument('--label_flag', type=str, default='N8', help='label nums: N8, D1,...,D20')
parser.add_argument('--nz', type=int, default=16, help='nz')
parser.add_argument('--layer', type=int, default=1, help='layer')
parser.add_argument('--loss', type=str, default='softmax', help='Apply loss')
parser.add_argument('--save_dir', type=str, default='checkpoints', help='save dir')
args = parser.parse_args()
print(args)

# set class
n_class = 1000

sys.stdout = Logger('log.txt')

# Normalize (0-1)
eps = args.eps/255.
use_gpu = torch.cuda.is_available()
if use_gpu:
    torch.backends.cudnn.benchmark = True
    torch.cuda.manual_seed_all(1111)
# GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Input dimensions
if args.model_type == 'res152':
    scale_size = 256
    img_size = 224
elif args.model_type == 'incv3':
    scale_size = 300
    img_size = 299

# Model
if args.model_type == 'incv3':
    model = torchvision.models.inception_v3(pretrained=True)
elif args.model_type == 'res152':
    model = torchvision.models.resnet152(pretrained=True)

if use_gpu:
    model = nn.DataParallel(model).cuda()
model.eval()

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

# Generator
if args.model_type == 'incv3':
    netG = ConGeneratorResnet(inception=True,nz=args.nz, layer=args.layer)
else:
    netG = ConGeneratorResnet(nz=args.nz, layer=args.layer)
netG = nn.DataParallel(netG).cuda()

# Optimizer
optimG = optim.Adam(netG.parameters(), lr=args.lr, betas=(0.5, 0.999))

# Data
data_transform = transforms.Compose([
    transforms.Resize(scale_size),
    transforms.CenterCrop(img_size),
    transforms.ToTensor(),
])

def normalize(t):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    t[:, 0, :, :] = (t[:, 0, :, :] - mean[0])/std[0]
    t[:, 1, :, :] = (t[:, 1, :, :] - mean[1])/std[1]
    t[:, 2, :, :] = (t[:, 2, :, :] - mean[2])/std[2]
    return t

train_dir = args.train_dir
train_set = datasets.ImageFolder(train_dir, data_transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
train_size = len(train_set)
print('Training data size:', train_size)

label_set = get_classes(args.label_flag)
# Loss
if args.loss == 'softmax':
    criterion = nn.CrossEntropyLoss()
elif args.loss == 'focal':
    criterion = FocalLoss()
else:
    raise IOError
####################
    
# Training
for epoch in range(args.epochs):
    running_loss = 0
    for i, (img, _) in enumerate(tqdm(train_loader)):
        img = img.cuda()
        np.random.shuffle(label_set)
        label = torch.from_numpy(np.random.choice(label_set, img.size(0))).long()
        z_class_one_hot = torch.zeros(img.size(0), n_class).scatter_(1, label.unsqueeze(1), 1).cuda()
        
        label = label.cuda()
        netG.train()
        optimG.zero_grad()
        # generate img
        if args.method == 'cond':
            noise = netG(input=img, z_one_hot=z_class_one_hot, eps=eps)
        adv = noise + img
        
        # the operation of tanh() has been completed in 'condgenerators.py'
        adv = torch.clamp(adv, 0.0, 1.0)

        loss = criterion(model(normalize(adv)), label)
        loss.backward()
        optimG.step()

        if i % 10 == 9:
            print('Epoch: {} \t Batch: {}/{} \t loss: {:.5f}'.format(epoch, i, len(train_loader), running_loss / 100))
            running_loss = 0
        running_loss += abs(loss.item())
        
    if epoch >= args.start_epoch:
        torch.save(netG.module.state_dict(), '{}/model-{}.pth'.format(args.save_dir, epoch))
