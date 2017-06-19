import argparse

import torch
import torch.nn as nn
from torch.autograd import Variable

from torchvision import datasets

parser = argparse.ArgumentParser(description='Image Recognition using Neural Networks')
parser.add_argument('--batch-size', type=int, default=20, metavar='N',
					help='batch size for training (default: 20)')
parser.add_argument('--folder-name', default='images', metavar='F',
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

# train_dataset = datasets.ImageFolder(root=args.folder_name)

