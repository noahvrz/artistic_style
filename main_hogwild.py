import argparse

import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.multiprocessing as mp

from torchvision import datasets, models, transforms
from torchvision.models.squeezenet import SqueezeNet, model_urls

INPUT_SIZE = 224
NUM_CLASSES = 25

parser = argparse.ArgumentParser(description='Image Recognition using Neural Networks')
parser.add_argument('--batch-size', type=int, default=20, metavar='N',
					help='batch size for training (default: 20)')
parser.add_argument('--num-processes', type=int, default=16, metavar='N',
					help='number of concurrent processes (default: 16)')
parser.add_argument('--folder-name', default='/mnt/research/gis/users/nverzani/ML/art_images', metavar='F',
					help='folder where images are stored (default: images)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                    help='learning rate (default: 0.0001)')
parser.add_argument('--resume', action='store_true', default=False,
                    help='resume from latest checkpoint')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--print-interval', type=int, default=100, metavar='N',
					help='interval at which to print training loss')

def worker(rank, args, model, start_epoch=1, best_accuracy=0):
	print(r)
	torch.manual_seed(args.seed + rank)

	# When testing, do some slight random scales and crops.
	# Normalize in the same way as the pre-trained Squeezenet.
	train_transforms = transforms.Compose([
		transforms.RandomSizedCrop(INPUT_SIZE),
		transforms.RandomHorizontalFlip(),
	    transforms.ToTensor(),
	    transforms.Normalize(mean=[0.485, 0.456, 0.406],
	                         std=[0.229, 0.224, 0.225])
	])

	# When testing and validating, just use center crop
	test_transforms = transforms.Compose([
		transforms.Scale(INPUT_SIZE),
		transforms.CenterCrop(256),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406],
	                         std=[0.229, 0.224, 0.225])
	])

	# The images are located in {train/val/test}/{style}/*.jpg
	train_dataset = datasets.ImageFolder(args.folder_name+"/train", train_transforms)
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

	val_dataset = datasets.ImageFolder(args.folder_name+"/val", test_transforms)
	val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)

	# test_dataset = datasets.ImageFolder(args.folder_name+"/test", test_transforms)
	# test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
	loss_function = nn.CrossEntropyLoss()

	if args.resume:
		save = torch.load('checkpoint.pth')
		optimizer.load_state_dict(save['optimizer'])

	for epoch in range(start_epoch, args.epochs + 1):
	    train(epoch, args, model, train_loader, optimizer, loss_function)
	    accuracy = val(epoch, val_loader)

	    if rank == 0:
		    save = {
		    	'epoch': epoch,
		    	'best_accuracy': best_accuracy,
		    	'model': model.state_dict(),
		    	'optimizer': optimizer.state_dict()
		    }

		    torch.save(save, 'checkpoint.pth')

		    if accuracy > best_accuracy:
		    	torch.save(save, 'best.pth')


def train(epoch, args, model, train_loader, optimizer, loss_function):
	model.train()
	pid = os.getpid()

	for i, (image, style) in enumerate(train_loader):

		image, style = Variable(image), Variable(style)

		optimizer.zero_grad()
		guess = model(image)
		loss = loss_function(guess, style)
		loss.backward()
		optimizer.step()

		if i % args.print_interval == 0:
			print("[Process {}][Epoch {}][Batch {}/{}]	Training loss: {}".format(
				pid, epoch, i, len(train_loader), loss.data[0]))

def val(epoch, val_loader):
	
	model.eval()

	loss = 0
	num_correct = 0
	for batch_num, (image, style) in enumerate(val_loader):

		image, style = Variable(image), Variable(style)

		guess = model(image)
		loss += loss_function(guess, style)
		
		prediction = guess.data.max(1)[1]
		num_correct += prediction.eq(style.data).cpu().sum()

	loss /= len(val_loader)
	accuracy = num_correct / len(val_dataset)

	print("[Epoch {}]	Average Testing Loss: {}, Accuracy: {}%".format(
		epoch, loss, 100*accuracy))

	return accuracy

if __name__ == '__main__':

	args = parser.parse_args()
	
	#os.environ['OMP_NUM_THREADS'] = '1'

	torch.manual_seed(args.seed)

	# Create a V1.1 Squeezenet, share its memory
	model = SqueezeNet(version=1.1, num_classes=NUM_CLASSES)
	model.share_memory()

	worker_args = (args, model)
	
	if args.resume:
		save = torch.load('checkpoint.pth')
		start_epoch = save['epoch']
		best_accuracy = save['best_accuracy']

		model.load_state_dict(save['model'])
		optimizer.load_state_dict(save['optimizer'])

		worker_args += (start_epoch, best_accuracy,)
	else:
		# Download a pretrained model, replace the last layer with a newly initialized one
		state_dict = torch.utils.model_zoo.load_url(model_urls['squeezenet1_1'])
		state_dict['classifier.1.weight'] = model.state_dict()['classifier.1.weight']
		state_dict['classifier.1.bias'] = model.state_dict()['classifier.1.bias']

		# Load the new parameters into the squeeze net
		model.load_state_dict(state_dict)

	# Start up the multiple processes
	processes = []
	for r in range(args.num_processes):
		p = mp.Process(target=worker, args=(r,)+worker_args)
		p.start()
		processes.append(p)
	for p in processes:
		p.join()
