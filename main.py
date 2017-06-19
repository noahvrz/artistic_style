import argparse

import torch
import torch.nn as nn
from torch.autograd import Variable

from torchvision import datasets, models

IMAGE_SIZE = 275
NUM_CLASSES = 25
NUM_CHANNELS = 3

parser = argparse.ArgumentParser(description='Image Recognition using Neural Networks')
parser.add_argument('--batch-size', type=int, default=20, metavar='N',
					help='batch size for training (default: 20)')
parser.add_argument('--folder-name', default='/mnt/research/gis/users/nverzani/ML/art_images', metavar='F',
					help='folder where images are stored (default: images)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

args = parser.parse_args()

torch.manual_seed(args.seed)
if args.cuda:
	torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

# train_dataset = datasets.ImageFolder(root=args.folder_name+"/train")
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=true, **kwargs)
# print(train_dataset.classes)
# print(train_dataset.class_to_idx)

# test_dataset = datasets.ImageFolder(root=args.folder_name+"/test")
# test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=true, **kwargs)
# val_dataset = datasets.ImageFolder(root=args.folder_name+"/val")
# val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=true, **kwargs)

kwargs = {'num_classes': NUM_CLASSES}
squeezenet = models.squeezenet1_1(pretrained=True, **kwargs)

#def train(epoch):

