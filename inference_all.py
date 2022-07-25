import argparse
import os
import numpy as np

import torchvision
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from condgenerators import ConGeneratorResnet
from utils import *
import torch.nn.functional as F
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Conditional Adversarial Generator')
parser.add_argument('--data_dir', default='data/ImageNet1k', help='ImageNet Validation Data')
parser.add_argument('--test_dir', default='', help='Testing Data')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size')
parser.add_argument('--model_t',type=str, default= 'res152',  help ='Model under attack : vgg16, vgg19, dense121' )
args = parser.parse_args()
print(args)

# Normalize (0-1)

n_class = 1000
# GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Input dimensions: Inception takes 3x299x299
if args.model_t in ['incv3', 'incv4']:
    img_size = 299
else:
    img_size = 224

model_t = load_model(args)
model_t = nn.DataParallel(model_t).cuda()
model_t.eval()

# Setup-Data
data_transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.ToTensor(),
])

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
def normalize(t):
    t[:, 0, :, :] = (t[:, 0, :, :] - mean[0])/std[0]
    t[:, 1, :, :] = (t[:, 1, :, :] - mean[1])/std[1]
    t[:, 2, :, :] = (t[:, 2, :, :] - mean[2])/std[2]

    return t

test_set = datasets.ImageFolder(args.test_dir, data_transform)
test_size = len(test_set)
print('Test data size:', test_size)

test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

# Evaluation
target_acc = 0.
target_test_size = 0.

for i, (img, _) in enumerate(tqdm(test_loader)):
    img = img.cuda()
    target_label_numpy = int(os.path.basename(test_set.samples[i][0]).split('_')[0][1:])
    adv_out = model_t(normalize(img.clone().detach())) 
    target_acc += torch.sum(adv_out.argmax(dim=-1) == target_label_numpy).item()
    print('{}\t{}'.format(i, target_acc))
    target_test_size += img.size(0)
    
print('target acc:{:.4%}\t target_test_size:{}'.format(target_acc/target_test_size, target_test_size))
