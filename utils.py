import torch
import torchvision
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import json
import sys
import os

class Logger(object):
    def __init__(self, fpath=None, mode='w'):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            self.file = open(fpath, mode)

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()

class FocalLoss(nn.Module):
    def __init__(self, gamma = 1, eps = 1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = nn.CrossEntropyLoss(reduce=False)

    def forward(self, input, target):
        logp = self.ce(input, target)
        print(logp)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        print((1 - p) ** self.gamma, loss)
        return loss.mean()

# Load ImageNet model to evaluate
def load_model(args):
    # Load Targeted Model
    if args.model_t == 'dense201':
        model_t = torchvision.models.densenet201(pretrained=True)
    elif args.model_t == 'vgg19':
        model_t = torchvision.models.vgg19(pretrained=True)
    elif args.model_t == 'vgg16':
        model_t = torchvision.models.vgg16(pretrained=True)
    elif args.model_t == 'googlenet':
        model_t = torchvision.models.googlenet(pretrained=True)
    return model_t

def fix_labels(args, test_set):
    val_dict = {}
    with open("val.txt") as file:
        for line in file:
            (key, val) = line.split(' ')
            val_dict[key.split('.')[0]] = int(val.strip())

    new_data_samples = []
    for i, j in enumerate(test_set.samples):
        org_label = val_dict[test_set.samples[i][0].split('/')[-1].split('.')[0]]
        new_data_samples.append((test_set.samples[i][0], org_label))

    test_set.samples = new_data_samples
    return test_set
############################################################

#############################################################
# This will fix labels for NIPS ImageNet
def fix_labels_nips(args, test_set, pytorch=False, target_flag=False):

    '''
    :param pytorch: pytorch models have 1000 labels as compared to tensorflow models with 1001 labels
    '''

    filenames = [i.split('/')[-1] for i, j in test_set.samples]
    # Load provided files and get image labels and names
    image_classes = pd.read_csv(os.path.join(args.data_dir, "images.csv"))
    image_metadata = pd.DataFrame({"ImageId": [f[:-4] for f in filenames]}).merge(image_classes, on="ImageId")
    true_classes = image_metadata["TrueLabel"].tolist()
    target_classes = image_metadata["TargetClass"].tolist()
    val_dict = {}
    for f, i in zip(filenames, range(len(filenames))):
        val_dict[f] = [true_classes[i], target_classes[i]]
    
    new_data_samples = []
    for i, j in enumerate(test_set.samples):
        if target_flag:
            org_label = val_dict[test_set.samples[i][0].split('/')[-1]][1]
        else:
            org_label = val_dict[test_set.samples[i][0].split('/')[-1]][0]
        if pytorch:
            new_data_samples.append((test_set.samples[i][0], org_label-1))
        else:
            new_data_samples.append((test_set.samples[i][0], org_label))

    test_set.samples = new_data_samples
    return test_set
