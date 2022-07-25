import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

import torchvision
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from condgenerators import ConGeneratorResnet
from utils import *
from get_class import get_classes

parser = argparse.ArgumentParser(description='Conditional Adversarial Generator')
parser.add_argument('--data_dir', default='data/ImageNet1k', help='ImageNet Validation Data')
parser.add_argument('--is_nips', action='store_true', default=True, help='Evaluation on NIPS data')
parser.add_argument('--batch_size', type=int, default=5, help='Batch Size')
parser.add_argument('--eps', type=int, default=16, help='Perturbation Budget')
parser.add_argument('--model_type', type=str, default= 'incv3',  help ='Model type incv3, res152')
parser.add_argument('--load_path', type=str, default='checkpoints/model.pth', help='load path')
parser.add_argument('--label_flag', type=str, default='N8', help='label nums: N8, D1,...,D20')
parser.add_argument('--nz', type=int, default=16, help='nz')
parser.add_argument('--layer', type=int, default=1, help='layer')
args = parser.parse_args()
print(args)

# Normalize (0-1)
eps = args.eps/255.
n_class = 1000
label_set = get_classes(args.label_flag)
print(label_set)

# GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Input dimensions
if args.model_type == 'res152':
    scale_size = 256
    img_size = 224
elif args.model_type == 'incv3':
    scale_size = 300
    img_size = 299

if args.model_type == 'incv3':
    netG = ConGeneratorResnet(inception=True, nz=args.nz, layer=args.layer)
else:
    netG = ConGeneratorResnet(nz=args.nz, layer=args.layer)

# Load Generator
netG.load_state_dict(torch.load(args.load_path))
netG = nn.DataParallel(netG).cuda()
netG.eval()

# Setup-Data
data_transform = transforms.Compose([
    transforms.Resize(scale_size),
    transforms.CenterCrop(img_size),
    transforms.ToTensor(),
])

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
def normalize(t):
    t[:, 0, :, :] = (t[:, 0, :, :] - mean[0])/std[0]
    t[:, 1, :, :] = (t[:, 1, :, :] - mean[1])/std[1]
    t[:, 2, :, :] = (t[:, 2, :, :] - mean[2])/std[2]
    return t

test_set = datasets.ImageFolder(args.data_dir, data_transform)
test_size = len(test_set)
print('Test data size:', test_size)

# Fix labels if needed
if args.is_nips:
    print('is_nips')
    test_set = fix_labels_nips(args, test_set, pytorch=True)
else:
    test_set = fix_labels(args, test_set)

test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
class_ids = np.array([150, 507, 62, 843, 426, 590, 715, 952])

# save
for idx in range(len(class_ids)):
    for i, (img, label) in enumerate(test_loader):
        img, label = img.cuda(), label.cuda()
        target_tensor = torch.LongTensor(img.size(0))
        target_tensor.fill_(class_ids[idx])
        target_one_hot = torch.zeros(img.size(0), n_class).scatter_(1, target_tensor.unsqueeze(1), 1).cuda()
        noise = netG(img, target_one_hot, eps=eps)
        adv = noise + img
        
        # Projection, tanh() have been operated in models.
        adv = torch.min(torch.max(adv, img - eps), img + eps)
        adv = torch.clamp(adv, 0.0, 1.0)

        save_imgs = adv.detach().cpu()
        for j in range(len(save_imgs)):
            g_img = transforms.ToPILImage('RGB')(save_imgs[j])
            output_dir = 'results/gan_n8/{}_t{}/images'.format(args.load_path.split('/')[-1].split('.pth')[0], class_ids[idx])
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            g_img.save(os.path.join(output_dir, '{}_{}.png'.format(class_ids[idx], i * args.batch_size + j)))
        
